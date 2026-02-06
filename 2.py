import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation 

# 执行指令
# uv run UniversalConverter.py -d ./imudata/TLIOv2/test -o ./imudata/TLIOv2/tlio_formatted/test -t tlio

# 引入基础模块
# 确保能找到 base，如果报错请保留这两行
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent
sys.path.append(str(project_root))

from base.datatype import GroundTruthData, ImuData
from base.serialize import UnitSerializer

class UniversalConverter:
    """Universal converter for various IMU dataset formats."""

    # Class-level constants
    G_TO_MS2 = 9.80665
    US_PER_SEC = 1_000_000
    MIN_SAMPLES_THRESHOLD = 10
    IDOL_MIN_FILE_SIZE_KB = 1
    QUAT_EPSILON = 1e-6
    QUAT_MIN_NORM = 0.1
    GRAVITY_MAGNITUDE_THRESHOLD = 8.0
    ACCELERATION_UNIT_G_MIN = 0.5
    ACCELERATION_UNIT_G_MAX = 1.5
    ACCELERATION_INVALID_THRESHOLD = 4.0

    # IDOL-specific settings
    IDOL_BUILDING_ROT_OFFSET = {
        "building1": 0.0,
        "building2": 1.8510,
        "building3": 0.2822
    }

    def __init__(self, dataset_root: str, output_root: str, dataset_type: str = "oxiod", freq: float = 100.0):
        """Initialize the UniversalConverter.

        Args:
            dataset_root: Input dataset root path
            output_root: Output root path
            dataset_type: Type of dataset (oxiod, ridi, rnin, idol, tlio, imunet, ronin)
            freq: Sampling frequency in Hz (default: 100.0)

        Raises:
            ValueError: If freq is not positive
        """
        if freq <= 0:
            raise ValueError(f"Invalid frequency: {freq}. Must be positive.")

        self.dataset_root = Path(dataset_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.dataset_type = dataset_type.lower()
        self.dt = 1.0 / freq

        # IDOL-specific settings (can be overridden)
        self.idol_remove_spike = True
        self.idol_spike_trunc_sec = 2
        self.idol_use_stencil_imu = True

    # ==========================================
    # Helper Methods
    # ==========================================

    @staticmethod
    def _normalize_timestamp(t_us: np.ndarray) -> np.ndarray:
        """Normalize timestamps to start from zero.

        Args:
            t_us: Timestamp array in microseconds

        Returns:
            Normalized timestamp array

        Raises:
            ValueError: If array is empty
        """
        if len(t_us) == 0:
            raise ValueError("Cannot normalize empty timestamp array")
        return t_us - t_us[0]

    @staticmethod
    def _align_ronin_by_gravity(acce: np.ndarray, gyro: np.ndarray,
                                gt_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Automatically detect and correct RoNIN IMU coordinate system misalignment.

        Principle: When static, IMU acceleration ≈ gravity. Search axis mapping
        to align IMU data with GT pose using gravity direction as reference.

        IMPROVED: Only uses static/low-motion data points for better accuracy.

        Args:
            acce: Raw acceleration (N, 3) in m/s^2
            gyro: Raw gyroscope (N, 3) in rad/s
            gt_quat: GT quaternion (N, 4) in xyzw format

        Returns:
            Tuple of aligned (acce, gyro, gt_quat)
        """
        from itertools import permutations

        # 1. Compute expected gravity direction in body frame
        # GT quaternion q_wb represents rotation from body to world
        # Gravity in world frame points DOWN: [0, 0, -g] or [0, 0, +g] depending on convention
        # To get gravity in body frame: g_body = R_wb^T * g_world

        # Try both conventions:
        # Convention A: Z-up, gravity = [0, 0, -1]
        # Convention B: Z-down, gravity = [0, 0, +1]
        g_world_convention_a = np.array([0, 0, -1.0])
        g_world_convention_b = np.array([0, 0, 1.0])

        rots = Rotation.from_quat(gt_quat)

        # Transform world gravity to body frame using inverse rotation (transpose)
        g_expected_a = rots.inv().apply(g_world_convention_a)
        g_expected_b = rots.inv().apply(g_world_convention_b)

        # We'll test both conventions and pick the one that gives better alignment
        # For now, use convention A (more common in robotics)
        g_expected = g_expected_a

        # 2. Downsample for faster search
        step = max(1, len(acce) // 1000)
        sub_acce = acce[::step]
        sub_g_exp = g_expected[::step]

        # 3. Filter static/low-motion points (IMPROVEMENT)
        # Only use points where acceleration magnitude is close to gravity
        acce_norm_mag = np.linalg.norm(sub_acce, axis=1)
        gravity_threshold = 9.81  # m/s^2
        tolerance = 1.5  # Allow ±1.5 m/s^2 deviation

        # Create mask for static/low-motion points
        static_mask = np.abs(acce_norm_mag - gravity_threshold) <= tolerance
        static_ratio = np.sum(static_mask) / len(static_mask)

        # Apply mask
        sub_acce_static = sub_acce[static_mask]
        sub_g_exp_static = sub_g_exp[static_mask]

        # Fallback: if too few static points, use all data
        if len(sub_acce_static) < 100:
            print(f"    → Warning: Only {len(sub_acce_static)} static points ({static_ratio*100:.1f}%), using all data")
            sub_acce_static = sub_acce
            sub_g_exp_static = sub_g_exp
        else:
            print(f"    → Using {len(sub_acce_static)} static points ({static_ratio*100:.1f}%)")

        # 4. Normalize acceleration (focus on direction only)
        acce_norm = sub_acce_static / (np.linalg.norm(sub_acce_static, axis=1, keepdims=True) + 1e-9)

        # 5. Search for best axis mapping
        axes = [0, 1, 2]
        signs = [1, -1]
        best_score = -1.0
        best_R = np.eye(3)
        best_label = "Identity"

        for p in permutations(axes):
            for sx in signs:
                for sy in signs:
                    for sz in signs:
                        # Build rotation matrix
                        R = np.zeros((3, 3))
                        R[0, p[0]] = sx
                        R[1, p[1]] = sy
                        R[2, p[2]] = sz
                        if np.linalg.det(R) < 0.5:  # Exclude reflection
                            continue

                        # Rotate acceleration
                        acce_rot = acce_norm @ R.T

                        # Compute correlation with expected gravity (dot product)
                        dot_prod = np.sum(acce_rot * sub_g_exp_static, axis=1)
                        score = np.mean(dot_prod)

                        if score > best_score:
                            best_score = score
                            best_R = R
                            best_label = f"Map:{p} Sign:({sx},{sy},{sz})"

        # 6. Apply best transformation
        acce_aligned = acce @ best_R.T
        gyro_aligned = gyro @ best_R.T

        print(f"    → Gravity alignment: score={best_score:.4f}, {best_label}")

        return acce_aligned, gyro_aligned, gt_quat

    def _extract_quat_safe(self, df: pd.DataFrame, prefix: str) -> np.ndarray | None:
        """Extract quaternion from DataFrame, validate it, and convert WXYZ to XYZW order.

        Args:
            df: Input DataFrame
            prefix: Column name prefix (e.g., 'gt_q', 'rv')

        Returns:
            Quaternion array in XYZW format, or None if invalid
        """
        cols = [f'{prefix}_w', f'{prefix}_x', f'{prefix}_y', f'{prefix}_z']
        if not all(c in df.columns for c in cols):
            return None

        # Extract raw data (W, X, Y, Z)
        q_raw = df[cols].values

        # Check if vector components are extremely small (likely static/invalid)
        # Only reject if all vector components are below threshold
        if np.all(np.abs(q_raw[:, 1:]) < self.QUAT_EPSILON):
            return None

        # Check norm, prevent all-zero data
        norms = np.linalg.norm(q_raw, axis=1)
        if np.mean(norms) < self.QUAT_MIN_NORM:
            return None

        # Normalize quaternions
        q_norm = q_raw / (norms[:, np.newaxis] + 1e-9)

        # Convert order: [w, x, y, z] -> [x, y, z, w]
        q_xyzw = np.zeros_like(q_norm)
        q_xyzw[:, 0:3] = q_norm[:, 1:4]  # Extract x, y, z to first three positions
        q_xyzw[:, 3] = q_norm[:, 0]  # Extract w to last position

        return q_xyzw

    # ==========================================
    # Dataset-specific Converters
    # ==========================================

    # ==========================================
    # 1. OxIOD Converter
    # ==========================================
    def parse_oxiod_imu(self, csv_path: Path) -> ImuData:
        """Parse OxIOD IMU data from CSV file.

        Args:
            csv_path: Path to IMU CSV file

        Returns:
            ImuData object containing parsed data

        Raises:
            ValueError: If file is empty
        """
        try:
            df = pd.read_csv(csv_path, header=None)
            if isinstance(df.iloc[0, 0], str):
                df = pd.read_csv(csv_path, header=0)
        except Exception:
            df = pd.read_csv(csv_path, header=None)

        count = len(df)
        if count == 0:
            raise ValueError(f"Empty IMU file: {csv_path}")

        t_us = np.arange(count, dtype=np.int64) * int(self.dt * self.US_PER_SEC)
        rpy = df.iloc[:, 1:4].values
        try:
            ahrs = Rotation.from_euler('xyz', rpy, degrees=False)
        except Exception:
            ahrs = Rotation.from_quat(np.array([0, 0, 0, 1]) * np.ones((count, 4)))

        gyro = df.iloc[:, 4:7].values
        gravity_g = df.iloc[:, 7:10].values
        user_acc_g = df.iloc[:, 10:13].values
        acce = (user_acc_g + gravity_g) * self.G_TO_MS2

        if df.shape[1] >= 16:
            magn = df.iloc[:, 13:16].values
        else:
            magn = np.zeros_like(acce)

        return ImuData(t_us, gyro, acce, ahrs, magn, frame="local")

    def parse_oxiod_gt(self, csv_path: Path) -> GroundTruthData:
        """Parse OxIOD ground truth data from CSV file.

        Args:
            csv_path: Path to GT CSV file

        Returns:
            GroundTruthData object containing parsed data

        Raises:
            ValueError: If file is empty
        """
        try:
            df = pd.read_csv(csv_path, header=None)
            if isinstance(df.iloc[0, 1], str):
                df = pd.read_csv(csv_path, header=0)
        except Exception:
            df = pd.read_csv(csv_path, header=None)

        count = len(df)
        if count == 0:
            raise ValueError(f"Empty GT file: {csv_path}")

        t_us = np.arange(count, dtype=np.int64) * int(self.dt * self.US_PER_SEC)
        ps = df.iloc[:, 2:5].values
        q_xyzw = df.iloc[:, 5:9].values
        rots = Rotation.from_quat(q_xyzw)
        return GroundTruthData(t_us, rots, ps)

    def convert_oxiod(self):
        """Convert OxIOD dataset to standard format."""
        print(f"--- Converting OxIOD from: {self.dataset_root} ---")
        search_pattern = str(self.dataset_root / "**" / "syn" / "imu*.csv")
        imu_files = glob.glob(search_pattern, recursive=True)

        if not imu_files:
            print(f"No OxIOD files found in {self.dataset_root}")
            return

        for imu_file in imu_files:
            imu_path = Path(imu_file)
            syn_dir = imu_path.parent
            file_id = imu_path.stem.replace("imu", "")
            gt_path = syn_dir / f"vi{file_id}.csv"

            if not gt_path.exists():
                gt_path_alt = syn_dir / "vi.csv"
                if gt_path_alt.exists():
                    gt_path = gt_path_alt
                else:
                    continue

            try:
                rel_path = syn_dir.relative_to(self.dataset_root).parent
            except Exception:
                rel_path = Path("unknown")

            try:
                print(f"Processing: {rel_path} / {imu_path.name}")
                imu_data = self.parse_oxiod_imu(imu_path)
                gt_data_raw = self.parse_oxiod_gt(gt_path)

                seq_name = imu_path.stem.replace("imu", "seq")
                if seq_name == "imu":
                    seq_name = "seq"

                save_dir = self.output_root / rel_path / seq_name
                UnitSerializer(imu_data, gt_data_raw).save(save_dir)
                print(f"  Saved to {save_dir}")

            except Exception as e:
                print(f"  Failed to process {imu_path.name}: {e}")

    # ==========================================
    # 2. RIDI Converter
    # ==========================================
    def convert_ridi(self):
        """Convert RIDI dataset to standard format."""
        print(f"--- Converting RIDI from: {self.dataset_root} ---")

        files = list(self.dataset_root.rglob("data.csv"))

        if not files:
            print(f"No 'data.csv' found.")
            return

        print(f"Found {len(files)} potential RIDI files.")

        for csv_path in files:
            if csv_path.parent.name == "processed":
                seq_name = csv_path.parent.parent.name
            else:
                seq_name = csv_path.parent.name

            print(f"Processing RIDI: {seq_name}")

            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()

                # IMU data parsing
                t_us = (df['time'].values / 1000.0).astype(np.int64)
                t_us = self._normalize_timestamp(t_us)

                gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values

                if 'acce_x' in df.columns:
                    acce = df[['acce_x', 'acce_y', 'acce_z']].values
                else:
                    acce = df[['linacce_x', 'linacce_y', 'linacce_z']].values

                if 'magnet_x' in df.columns:
                    magn = df[['magnet_x', 'magnet_y', 'magnet_z']].values
                else:
                    magn = np.zeros_like(gyro)

                # Parse rotation vector as AHRS
                if 'rv_w' in df.columns:
                    # Scipy Rotation expects [x, y, z, w] order
                    rv_xyzw = df[['rv_x', 'rv_y', 'rv_z', 'rv_w']].values
                    ahrs = Rotation.from_quat(rv_xyzw)
                else:
                    print("  Warning: 'rv' columns not found, using Identity.")
                    ahrs = Rotation.identity(len(t_us))

                imu_data = ImuData(t_us, gyro, acce, ahrs, magn)

                # Ground truth parsing
                pos = df[['pos_x', 'pos_y', 'pos_z']].values

                # GT orientation (ori)
                if 'ori_w' in df.columns:
                    quat_xyzw = df[['ori_x', 'ori_y', 'ori_z', 'ori_w']].values
                else:
                    quat_xyzw = df.iloc[:, -4:].values

                gt_data = GroundTruthData(t_us, Rotation.from_quat(quat_xyzw), pos)

                # Save
                save_dir = self.output_root / seq_name
                UnitSerializer(imu_data, gt_data).save(save_dir)
                print(f"  Saved to {save_dir}")

            except Exception as e:
                print(f"  Error: Failed to convert {seq_name}: {e}")

    # ==========================================
    # 3. RNIN (SenseINS) Converter
    # ==========================================
    def convert_rnin(self):
        """Convert RNIN (SenseINS) dataset to standard format."""
        print(f"--- Converting RNIN (SenseINS) from: {self.dataset_root} ---")

        files = list(self.dataset_root.rglob("*.csv"))
        if not files:
            print(f"No CSV files found in {self.dataset_root}")
            return

        print(f"Found {len(files)} potential RNIN files.")

        for csv_path in files:
            # Optimize sequence naming: if file name is SenseINS, use parent folder name
            seq_name = csv_path.stem
            if seq_name.lower() in ["data", "imu", "synced", "senseins"]:
                seq_name = csv_path.parent.name

            print(f"Processing RNIN: {seq_name}")

            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip().str.lower()

                # Timestamp parsing (s -> us)
                time_col = 'times' if 'times' in df.columns else 'time'
                if time_col not in df.columns:
                    print(f"  Skip: No time column found in {seq_name}")
                    continue
                t_us = (df[time_col].values * self.US_PER_SEC).astype(np.int64)
                t_us = self._normalize_timestamp(t_us)

                # IMU data parsing
                gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values
                acce = df[['acce_x', 'acce_y', 'acce_z']].values

                # AHRS orientation extraction (prioritize rv, then gv)
                ahrs_xyzw = self._extract_quat_safe(df, 'rv')
                if ahrs_xyzw is None:
                    ahrs_xyzw = self._extract_quat_safe(df, 'gv')

                if ahrs_xyzw is not None:
                    ahrs = Rotation.from_quat(ahrs_xyzw)
                else:
                    print("  Warning: No valid RV/GV AHRS, using Identity.")
                    ahrs = Rotation.identity(len(t_us))

                # 4. 磁力计
                if 'magnet_un_x' in df.columns:
                    magn = df[['magnet_un_x', 'magnet_un_y', 'magnet_un_z']].values
                else:
                    magn = np.zeros_like(gyro)

                imu_data = ImuData(t_us, gyro, acce, ahrs, magn, frame="local")
                
                # 5. Ground Truth (GT) 提取
                # 5.1 位置 (Position)
                # 对于 RNIN 数据集，gt_p 通常是全 0（GNSS/RTK 仅有位置无姿态），直接使用 vio_p
                pos = df[['vio_p_x', 'vio_p_y', 'vio_p_z']].values
                print("  -> [Info] Using vio_p as position ground truth (RNIN standard)")

                # 5.2 姿态 (Orientation)
                # 对于 RNIN 数据集，优先级: vio_q -> rv -> gt_q -> identity
                # 原因: gt_q 通常是单位四元数（无姿态信息），vio_q 包含真实的 VIO 姿态
                gt_xyzw = self._extract_quat_safe(df, 'vio_q')
                if gt_xyzw is not None:
                    print("  -> [Info] Using vio_q as orientation ground truth")
                else:
                    print("  -> [Info] vio_q not available, trying rv...")
                    gt_xyzw = self._extract_quat_safe(df, 'rv')
                    if gt_xyzw is not None:
                        print("  -> [Info] Using rv as orientation ground truth")
                    else:
                        print("  -> [Warning] No VIO orientation found, falling back to gt_q...")
                        gt_xyzw = self._extract_quat_safe(df, 'gt_q')
                        if gt_xyzw is None:
                            print("  -> [Warning] All orientation sources unavailable, using identity!")
                            gt_xyzw = np.tile([0, 0, 0, 1], (len(t_us), 1))
                
                gt_data = GroundTruthData(t_us, Rotation.from_quat(gt_xyzw), pos)
                
                # 6. 保存数据
                save_dir = self.output_root / seq_name
                UnitSerializer(imu_data, gt_data).save(save_dir)
                print(f"  -> Saved to {save_dir} ({len(t_us)} samples)")
                
            except Exception as e:
                print(f"  [Error] Failed to convert {seq_name}: {e}")

    # ==========================================
    # 4. IDOL Converter
    # ==========================================
    def _parse_idol_feather(self, feather_path: Path, building: str) -> tuple[ImuData, GroundTruthData]:
        """Parse single IDOL .feather file and return IMU and GT data objects.

        Args:
            feather_path: Path to the feather file
            building: Building name for rotation offset

        Returns:
            Tuple of (ImuData, GroundTruthData)

        Raises:
            ValueError: If required columns are missing or no data left after spike removal
        """
        # Read feather file
        df = pd.read_feather(feather_path)

        # Column mapping (original column name -> internal column name)
        col_mapping = {
            # Core fields
            'timestamp': 'timestamp',
            # GT orientation (quaternion wxyz)
            'orientW': 'orient_w',
            'orientX': 'orient_x',
            'orientY': 'orient_y',
            'orientZ': 'orient_z',
            # iPhone orientation (backup)
            'iphoneOrientW': 'iphone_orient_w',
            'iphoneOrientX': 'iphone_orient_x',
            'iphoneOrientY': 'iphone_orient_y',
            'iphoneOrientZ': 'iphone_orient_z',
            # Stencil IMU (consistent with GT reference frame)
            'stencilGyroX': 'stencil_gyro_x',
            'stencilGyroY': 'stencil_gyro_y',
            'stencilGyroZ': 'stencil_gyro_z',
            'stencilAccX': 'stencil_acc_x',
            'stencilAccY': 'stencil_acc_y',
            'stencilAccZ': 'stencil_acc_z',
            # iPhone IMU
            'iphoneGyroX': 'iphone_gyro_x',
            'iphoneGyroY': 'iphone_gyro_y',
            'iphoneGyroZ': 'iphone_gyro_z',
            'iphoneAccX': 'iphone_acc_x',
            'iphoneAccY': 'iphone_acc_y',
            'iphoneAccZ': 'iphone_acc_z',
            # iPhone magnetometer
            'iphoneMagX': 'iphone_mag_x',
            'iphoneMagY': 'iphone_mag_y',
            'iphoneMagZ': 'iphone_mag_z',
            # GT position
            'processedPosX': 'processed_pos_x',
            'processedPosY': 'processed_pos_y',
            'processedPosZ': 'processed_pos_z'
        }

        # Rename columns
        df = df.rename(columns=col_mapping)

        # Ensure all required columns exist
        required_cols = ['timestamp', 'orient_w', 'orient_x', 'orient_y', 'orient_z',
                        'processed_pos_x', 'processed_pos_y', 'processed_pos_z']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col} (column mapping failed)")

        # Remove sync spikes (first and last N seconds)
        if self.idol_remove_spike:
            start_ts = df["timestamp"].min()
            end_ts = df["timestamp"].max()
            df = df[
                (df["timestamp"] >= start_ts + self.idol_spike_trunc_sec) &
                (df["timestamp"] <= end_ts - self.idol_spike_trunc_sec)
            ]
            if len(df) == 0:
                raise ValueError("No data left after removing sync spike")

        # Timestamp conversion: seconds -> microseconds, normalize to 0
        t_sec = df["timestamp"].values
        t_us = (t_sec * self.US_PER_SEC).astype(np.uint64)
        t_us = self._normalize_timestamp(t_us)

        # IMU data extraction + reference frame conversion
        if self.idol_use_stencil_imu:
            # Prefer Stencil IMU (consistent with GT reference frame)
            gyro = df[["stencil_gyro_x", "stencil_gyro_y", "stencil_gyro_z"]].values
            acce = df[["stencil_acc_x", "stencil_acc_y", "stencil_acc_z"]].values
        else:
            # iPhone IMU: reference frame conversion (Stencil +x->iPhone -x, +y->-y, +z->+z)
            gyro = df[["iphone_gyro_x", "iphone_gyro_y", "iphone_gyro_z"]].values
            gyro[:, 0] = -gyro[:, 0]
            gyro[:, 1] = -gyro[:, 1]

            acce = df[["iphone_acc_x", "iphone_acc_y", "iphone_acc_z"]].values
            acce[:, 0] = -acce[:, 0]
            acce[:, 1] = -acce[:, 1]

        # Magnetometer (only from iPhone)
        if all(col in df.columns for col in ["iphone_mag_x", "iphone_mag_y", "iphone_mag_z"]):
            magn = df[["iphone_mag_x", "iphone_mag_y", "iphone_mag_z"]].values
        else:
            magn = np.zeros_like(gyro)

        # AHRS orientation: prefer Stencil GT orientation
        ahrs_xyzw = self._extract_quat_safe(df, 'orient')
        if ahrs_xyzw is None:
            ahrs_xyzw = self._extract_quat_safe(df, 'iphone_orient')
        
        if ahrs_xyzw is not None:
            ahrs = Rotation.from_quat(ahrs_xyzw)
        else:
            ahrs = Rotation.identity(len(t_us))
        
        imu_data = ImuData(t_us, gyro, acce, ahrs, magn, frame="local")
        
        # GT data extraction
        # Position: processed_pos_x/y/z (already smoothed)
        pos = df[["processed_pos_x", "processed_pos_y", "processed_pos_z"]].values

        # Apply building rotation offset (counterclockwise rotation in XY plane)
        rot_offset = self.IDOL_BUILDING_ROT_OFFSET.get(building.lower(), 0.0)
        if rot_offset != 0.0:
            cos_rot = np.cos(rot_offset)
            sin_rot = np.sin(rot_offset)
            rot_mat = np.array([
                [cos_rot, -sin_rot, 0],
                [sin_rot, cos_rot, 0],
                [0, 0, 1]
            ])
            pos = pos @ rot_mat.T

        # GT orientation: force read orient quaternion (skip strict validation to ensure value exists)
        # Manually extract wxyz quaternion to avoid _extract_quat_safe misjudgment
        q_wxyz = df[["orient_w", "orient_x", "orient_y", "orient_z"]].values
        # Normalize quaternion (must do this)
        q_norm = np.linalg.norm(q_wxyz, axis=1)
        q_norm[q_norm < self.QUAT_EPSILON] = 1.0  # Avoid division by zero
        q_wxyz = q_wxyz / q_norm[:, np.newaxis]
        # Convert to xyzw format (for Scipy compatibility)
        gt_xyzw = np.zeros_like(q_wxyz)
        gt_xyzw[:, 0:3] = q_wxyz[:, 1:4]  # x,y,z
        gt_xyzw[:, 3] = q_wxyz[:, 0]  # w

        gt_rots = Rotation.from_quat(gt_xyzw)
        gt_data = GroundTruthData(t_us, gt_rots, pos)

        return imu_data, gt_data

    def convert_idol(self):
        """Convert IDOL dataset to standard format."""
        print(f"--- Converting IDOL from: {self.dataset_root} ---")

        # Building/subset identification logic
        if "building" in self.dataset_root.name.lower():
            building_dirs = [self.dataset_root]
        else:
            building_dirs = [d for d in self.dataset_root.iterdir()
                          if d.is_dir() and "building" in d.name.lower()]

        if not building_dirs:
            building_name = self.dataset_root.parent.name
            subset_dirs = [self.dataset_root]
            print(f"Warning: No building folder found, treating as subset directory: {self.dataset_root} (building: {building_name})")
        else:
            for building_dir in building_dirs:
                building_name = building_dir.name
                subset_dirs = [d for d in building_dir.iterdir() if d.is_dir()]

        # Process subset directories
        for subset_dir in subset_dirs:
            subset_name = subset_dir.name
            print(f"\n-- Processing Subset: {subset_name} --")

            meta_path = subset_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                print(f"  Found {len(meta)} trajectories in metadata")

            feather_files = list(subset_dir.glob("*.feather"))
            if not feather_files:
                print(f"  No .feather files found in {subset_dir}")
                continue

            # Process all feather files
            for feather_file in feather_files:
                seq_name = feather_file.stem
                print(f"\n===== Processing IDOL: {building_name}/{subset_name}/{seq_name} =====")

                try:
                    # Check feather file size (skip empty files)
                    if feather_file.stat().st_size < self.IDOL_MIN_FILE_SIZE_KB * 1024:
                        print(f"  Skip {seq_name}: Empty feather file (size < {self.IDOL_MIN_FILE_SIZE_KB}KB)")
                        continue

                    # Parse feather file
                    imu_data, gt_data = self._parse_idol_feather(feather_file, building_name)

                    # Check data count (skip if no data)
                    if len(imu_data.t_us) < self.MIN_SAMPLES_THRESHOLD:
                        print(f"  Skip {seq_name}: Too few samples ({len(imu_data.t_us)} frames)")
                        continue

                    # Save data
                    save_dir = self.output_root / seq_name
                    UnitSerializer(imu_data, gt_data).save(save_dir)
                    print(f"  Saved to {save_dir} ({len(imu_data.t_us)} samples)")

                except Exception as e:
                    print(f"  Failed to convert {seq_name}: {str(e)[:200]}")
    

    # ==========================================
    # 5. TLIO Converter
    # ==========================================
    def convert_tlio(self):
        """Convert TLIO dataset to standard format."""
        print(f"--- Converting TLIO from: {self.dataset_root} ---")
        npy_files = list(self.dataset_root.rglob("imu0_resampled.npy"))
        if not npy_files:
            print("  No imu0_resampled.npy files found.")
            return

        for npy_path in npy_files:
            rel_dir = npy_path.parent.relative_to(self.dataset_root)
            print(f"\n>>> Processing TLIO: {rel_dir}")

            try:
                raw_data = np.load(npy_path)
                num_features = raw_data.shape[-1]
                data = raw_data.reshape(-1, num_features)

                print(f"  Info: Data shape: {data.shape}")

                # Core fix: for your data format [ts, gyr, acc, quat, pos, vel]
                # Your data should have 1+3+3+4+3+3 = 17 columns
                if data.shape[1] == 17:
                    # Extract each part (Python slicing is left-closed, right-open)
                    # col 0: ts_us
                    t_data = data[:, 0:1]
                    # col 1-4: gyr (3)
                    gyro = data[:, 1:4]
                    # col 4-7: acc (3)
                    acce = data[:, 4:7]
                    # col 7-11: qxyzw (4) - Note: quaternion comes before position
                    quat_raw = data[:, 7:11]
                    # col 11-14: pos (3)
                    pos = data[:, 11:14]

                    print("  Format: Detected custom format: [ts, gyr, acc, quat, pos, vel]")
                else:
                    # Fallback logic (for other format files)
                    print(f"  Warning: Unexpected column count {data.shape[1]}. Trying standard mapping...")
                    t_data = data[:, 0:1]
                    gyro = data[:, 1:4]
                    acce = data[:, 4:7]
                    pos = data[:, 7:10]  # Standard TLIO usually has Pos first
                    quat_raw = data[:, 10:14]

                # Time unit processing
                t_val = t_data.flatten()
                t0_raw = t_val[0]

                # Smart unit detection
                if t0_raw > 1e16:  # Nanoseconds
                    print(f"  Time: Detected Nanoseconds. Converting to us.")
                    t_us = (t_val / 1000).astype(np.int64)
                elif t0_raw > 1e12:  # Microseconds (~16 digits)
                    print(f"  Time: Detected Microseconds. Keeping as is.")
                    t_us = t_val.astype(np.int64)
                else:  # Seconds
                    print(f"  Time: Detected Seconds. Converting to us.")
                    t_us = (t_val * self.US_PER_SEC).astype(np.int64)

                t_us = self._normalize_timestamp(t_us)

                # Quaternion normalization and construction
                # Ensure quaternion is valid
                if quat_raw.shape[1] != 4:
                    raise ValueError(f"Quaternion column shape wrong: {quat_raw.shape}")

                norms = np.linalg.norm(quat_raw, axis=1, keepdims=True)
                quat_raw = quat_raw / (norms + 1e-9)
                rots = Rotation.from_quat(quat_raw)

                # Save
                imu_data = ImuData(t_us, gyro, acce, ahrs=rots, magn=np.zeros_like(gyro))
                gt_data = GroundTruthData(t_us, rots, pos)

                save_dir = self.output_root / rel_dir
                UnitSerializer(imu_data, gt_data).save(save_dir)
                print(f"  Saved to {save_dir} (N={len(t_us)})")

            except Exception as e:
                print(f"  ❌ [Error] {rel_dir} Processing Failed: {e}")
    
    # ==========================================
    # IMUnet 处理逻辑
    # ==========================================

    def _convert_timestamp_ns_to_us(self, time_values: np.ndarray, seq_name: str) -> np.ndarray:
        """将纳秒时间戳转换为微秒（IMUNet专用）

        Args:
            time_values: 原始时间戳数组（纳秒）
            seq_name: 序列名称（用于日志）

        Returns:
            转换后的时间戳（微秒）
        """
        if len(time_values) < 2:
            print(f"  Warning: {seq_name} has less than 2 timestamps")
            return (time_values / 1000.0).astype(np.int64)

        # 纳秒转微秒：除以1000
        t_us = (time_values / 1000.0).astype(np.int64)

        # 验证转换后的采样周期
        converted_diff = np.median(np.diff(t_us)) / 1e6  # 转换为秒
        print(f"  [Timestamp] Converted NS to US: sampling period = {converted_diff*1000:.2f}ms ({converted_diff:.6f}s)")

        if converted_diff < 0.001 or converted_diff > 1.0:
            print(f"  [Timestamp] WARNING: Abnormal sampling period! Expected 5-200ms, got {converted_diff*1000:.2f}ms")

        return t_us

    def convert_imunet(self):
        """Convert IMUnet dataset to standard format."""
        print(f"--- Converting IMUnet from: {self.dataset_root} ---")

        files = list(self.dataset_root.rglob("*.csv"))
        if not files:
            print(f"No CSV files found in {self.dataset_root}")
            return

        files = [f for f in files if "readme" not in f.stem.lower()]
        print(f"Found {len(files)} potential IMUnet files.")

        for csv_path in files:
            # Sequence naming logic
            generic_names = {"data", "processed", "imu", "synced", "output", "log", "csv", "data_plain"}
            current_path = csv_path
            seq_name = current_path.stem
            while seq_name.lower() in generic_names:
                current_path = current_path.parent
                seq_name = current_path.name
                if current_path == self.dataset_root:
                    break

            print(f"Processing IMUnet: {seq_name}")

            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip().str.lower()

                # Timestamp - IMUNet的时间戳是纳秒，需要转换为微秒
                if 'time' not in df.columns:
                    print(f"  Skip: No 'time' column.")
                    continue

                # 纳秒转微秒：除以1000
                t_raw = df['time'].values
                t_us = self._convert_timestamp_ns_to_us(t_raw, seq_name)
                t_us = self._normalize_timestamp(t_us)
                print(f"  Time: Normalized. Duration: {t_us[-1] / self.US_PER_SEC:.2f}s, Samples: {len(t_us)}")

                # IMU data extraction
                gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values

                # --- Smart acceleration extraction logic ---
                # A. Try to read base acce
                if 'acce_x' in df.columns:
                    base_acce = df[['acce_x', 'acce_y', 'acce_z']].values
                else:
                    print("  Error: No 'acce' column found.")
                    continue

                # B. Calculate base acce norm to determine what it is
                avg_norm = np.mean(np.linalg.norm(base_acce, axis=1))
                print(f"  Check: Base Acce Mean Norm: {avg_norm:.4f}")

                final_acce = None

                # Case 1: Acce is already Raw (norm ~9.8) -> use directly
                if avg_norm > self.GRAVITY_MAGNITUDE_THRESHOLD:
                    print("  Info: 'acce' column is already Raw Acceleration (contains gravity). Using directly.")
                    final_acce = base_acce

                # Case 2: Acce is Linear (norm is small), need to compose
                else:
                    # Look for gravity column (compatible with gravity_x and grav_x)
                    grav_cols = None
                    if 'gravity_x' in df.columns:
                        grav_cols = ['gravity_x', 'gravity_y', 'gravity_z']
                    elif 'grav_x' in df.columns:
                        grav_cols = ['grav_x', 'grav_y', 'grav_z']

                    if grav_cols:
                        grav_vec = df[grav_cols].values
                        # Check unit: if it's g (norm~1.0) need to convert to m/s^2
                        base_is_g = avg_norm < 2.0 and avg_norm > 0.5

                        # Execute composition: Raw = Linear + Gravity
                        final_acce = base_acce + grav_vec
                        print(f"  Info: Reconstructed Raw Accel (Linear + {grav_cols[0].split('_')[0]}).")
                    else:
                        # No gravity column, try unit scaling only
                        print("  Warning: Low magnitude accel found but NO gravity column. Attempting unit scaling only.")
                        final_acce = base_acce

                # C. Final unit check (handle case where unit is g)
                # Check final result norm
                final_norm = np.mean(np.linalg.norm(final_acce, axis=1))

                if self.ACCELERATION_UNIT_G_MIN < final_norm < self.ACCELERATION_UNIT_G_MAX:
                    print(f"  Unit: Final data looks like 'g' (norm={final_norm:.2f}). Scaling by {self.G_TO_MS2}.")
                    final_acce *= self.G_TO_MS2
                elif final_norm < self.ACCELERATION_INVALID_THRESHOLD:
                    print(f"  CRITICAL WARNING: Final Accel norm is still too low ({final_norm:.2f}). Data is likely invalid!")

                acce = final_acce

                # Magnetometer
                if 'magnet_x' in df.columns:
                    magn = df[['magnet_x', 'magnet_y', 'magnet_z']].values
                else:
                    magn = np.zeros_like(gyro)

                # AHRS (RV) orientation
                if 'rv_w' in df.columns:
                    rv_xyzw = df[['rv_x', 'rv_y', 'rv_z', 'rv_w']].values
                    ahrs = Rotation.from_quat(rv_xyzw)
                else:
                    ahrs = Rotation.identity(len(t_us))

                imu_data = ImuData(t_us, gyro, acce, ahrs, magn, frame="local")

                # Ground Truth
                if 'pos_x' in df.columns:
                    pos = df[['pos_x', 'pos_y', 'pos_z']].values
                else:
                    pos = np.zeros_like(gyro)

                if 'ori_w' in df.columns:
                    gt_xyzw = df[['ori_x', 'ori_y', 'ori_z', 'ori_w']].values
                    gt_rot = Rotation.from_quat(gt_xyzw)
                else:
                    gt_rot = Rotation.identity(len(t_us))

                gt_data = GroundTruthData(t_us, gt_rot, pos)

                save_dir = self.output_root / seq_name
                UnitSerializer(imu_data, gt_data).save(save_dir)
                print(f"  Saved to {save_dir} ({len(t_us)} samples)")

            except Exception as e:
                print(f"  Error: Failed to convert {seq_name}: {e}")

    # ==========================================
    # 7. RoNIN Converter
    # ==========================================
    def convert_ronin(self):
        """Convert RoNIN dataset to standard format."""
        print(f"--- Converting RoNIN from: {self.dataset_root} ---")
        
        # Find all .hdf5 files
        files = list(self.dataset_root.rglob("*.hdf5"))
        if not files:
            print(f"No HDF5 files found in {self.dataset_root}")
            return

        print(f"Found {len(files)} RoNIN HDF5 files.")

        for h5_path in files:
            # Construct relative path to preserve directory structure
            rel_dir = h5_path.parent.relative_to(self.dataset_root)
            seq_name = str(rel_dir).replace("/", "_")  # Optional: flatten path as sequence name

            print(f"\n>>> Processing RoNIN: {rel_dir}")

            try:
                with h5py.File(h5_path, 'r') as f:
                    # 1. Check necessary data groups
                    if 'synced' not in f or 'pose' not in f:
                        print(f"  Skip: Missing 'synced' or 'pose' group in {h5_path.name}")
                        continue

                    # 2. Read Synced data (IMU Input)
                    # RoNIN synced data is already 200Hz aligned
                    synced = f['synced']

                    acce = synced['acce'][:]      # (N, 3) m/s^2
                    gyro = synced['gyro'][:]      # (N, 3) rad/s
                    time_s = synced['time'][:]    # (N, ) seconds

                    # Read orientation for AHRS (use game_rv which is fusion without magnetometer)
                    if 'game_rv' in synced:
                        game_rv = synced['game_rv'][:]  # (N, 4) [x, y, z, w]
                    elif 'rv' in synced:
                        game_rv = synced['rv'][:]
                    else:
                        # If no orientation, initialize to unit quaternion
                        game_rv = np.array([[0, 0, 0, 1]] * len(time_s))

                    # RoNIN typically doesn't emphasize magnetometer, use zero padding
                    if 'magnet' in synced:
                        magn = synced['magnet'][:]
                    else:
                        magn = np.zeros_like(gyro)

                    # 3. Read Pose data (Ground Truth)
                    pose = f['pose']

                    # Prefer Tango ground truth
                    if 'tango_pos' in pose:
                        gt_pos = pose['tango_pos'][:]  # (N, 3) meters
                    else:
                        print("  Warning: No tango_pos, looking for alternatives...")
                        gt_pos = np.zeros_like(acce)

                    if 'tango_ori' in pose:
                        gt_quat_raw = pose['tango_ori'][:]  # (N, 4) [x, y, z, w]
                    elif 'ekf_ori' in pose:
                        gt_quat_raw = pose['ekf_ori'][:]
                    else:
                        gt_quat_raw = np.array([[0, 0, 0, 1]] * len(time_s))

                    # 4. NOTE: align_tango_to_body is NOT applied here
                    # Reason: Based on testing, applying align_tango_to_body does NOT correctly align
                    # GT orientation to IMU body frame (results in 98° error from gravity)
                    # Instead, GT orientation is kept in Tango frame, and alignment is applied
                    # during evaluation in StandardEvaluator
                    info_path = h5_path.parent / "info.json"
                    if info_path.exists():
                        try:
                            with open(info_path, 'r') as info_file:
                                info_data = json.load(info_file)

                            if 'align_tango_to_body' in info_data:
                                # Read but DO NOT apply - save for evaluation
                                align_q = np.array(info_data['align_tango_to_body'])
                                R_align = Rotation.from_quat(align_q)
                                euler = R_align.as_euler('xyz', degrees=True)
                                print(f"  [INFO] Found align_tango_to_body: Euler={np.linalg.norm(euler):.1f}deg")
                                print(f"  [INFO] GT orientation kept in Tango frame (alignment applied in evaluation)")
                                # Keep GT in original Tango frame (don't apply alignment)
                                gt_quat = gt_quat_raw
                            else:
                                # No alignment info
                                gt_quat = gt_quat_raw
                                print("  [ALIGN] No align_tango_to_body found, using raw GT")
                        except Exception as e:
                            print(f"  [WARNING] Failed to read info.json: {e}")
                            gt_quat = gt_quat_raw
                    else:
                        # No info.json, use raw quaternions
                        gt_quat = gt_quat_raw
                        print("  [ALIGN] No info.json found, using raw GT")

                    # 5. Data preprocessing

                    # A. Timestamp conversion: seconds -> microseconds (int64)
                    t_us = (time_s * self.US_PER_SEC).astype(np.int64)

                    # B. Force time to start from zero
                    start_offset = t_us[0]
                    t_us = self._normalize_timestamp(t_us)
                    print(f"  Time: Normalized start. Range: {t_us[-1] / self.US_PER_SEC:.2f}s")

                    # C. Verify row alignment
                    n_samples = len(t_us)
                    if not (len(acce) == len(gt_pos) == n_samples):
                        print(f"  Error: Length mismatch! Time:{n_samples}, Acce:{len(acce)}, GT:{len(gt_pos)}")
                        continue

                    # D. RoNIN coordinate system alignment
                    # NOTE: align_tango_to_body is NOT applied (kept in Tango frame)
                    # Alignment will be applied during evaluation in StandardEvaluator
                    print("  [INFO] GT in Tango frame (alignment applied in evaluation)")


                    # 5. Build objects

                    # Build AHRS rotation object (RoNIN quaternion order is scalar-last: x, y, z, w)
                    # Scipy also expects x, y, z, w
                    ahrs_rot = Rotation.from_quat(game_rv)
                    gt_rot = Rotation.from_quat(gt_quat)

                    # Package IMU data
                    imu_data = ImuData(
                        t_us=t_us,
                        gyro=gyro,
                        acce=acce,
                        ahrs=ahrs_rot,
                        magn=magn
                    )

                    # Package GT data
                    gt_data = GroundTruthData(t_us, gt_rot, gt_pos)

                    # 6. Save
                    # Preserve original directory structure
                    save_dir = self.output_root / rel_dir
                    UnitSerializer(imu_data, gt_data).save(save_dir)
                    print(f"  Saved to {save_dir} (N={n_samples})")

            except Exception as e:
                print(f"  Error: Failed to process {rel_dir}: {e}")
    # 主执行函数（新增idol分支）
    # ==========================================
    def run(self):
        if self.dataset_type == "oxiod":
            self.convert_oxiod()
        elif self.dataset_type == "ridi":
            self.convert_ridi()
        elif self.dataset_type == "rnin":
            self.convert_rnin()
        elif self.dataset_type == "idol":  # 新增IDOL分支
            self.convert_idol()
        elif self.dataset_type == "tlio":  # 新增IDOL分支
            self.convert_tlio()
        elif self.dataset_type == "imunet":  # 新增IDOL分支
            self.convert_imunet()
        elif self.dataset_type == "ronin":  # 新增IDOL分支
            self.convert_ronin()
        else:
            print(f"Unknown dataset type: {self.dataset_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert datasets to standard format")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Input dataset root path")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output root path")
    parser.add_argument("-t", "--type", type=str, default="oxiod", choices=["oxiod", "ridi", "rnin","idol","tlio","imunet","ronin"], help="Dataset type")
    
    args = parser.parse_args()
    
    converter = UniversalConverter(args.dataset, args.output, dataset_type=args.type, freq=100.0)
    converter.run()