from pathlib import Path
from SIMDAdapter import SIMDAdapter
from base.serialize import ImuDataSerializer, PosesDataSerializer

# 测试加载单个文件
csv_path = Path('/home/vln/imuproject/MdlVerifyV1/imudata/SIMD/all/2021-12-11 17-36-39_user22.csv')
unit = SIMDAdapter.load(csv_path)

print(f'✅ Loaded: {unit.name}')
print(f'  IMU samples: {len(unit.imu_data.t_us)}')
print(f'  GT samples: {len(unit.gt_data.t_us)}')
print(f'  Duration: {unit.imu_data.t_us[-1] / 1e6 - unit.imu_data.t_us[0] / 1e6:.2f} seconds')
print(f'  Gyro shape: {unit.imu_data.gyro.shape}')
print(f'  Acce shape: {unit.imu_data.acce.shape}')
print(f'  AHRS type: {type(unit.imu_data.ahrs)}')
print(f'  GT rots type: {type(unit.gt_data.rots)}')

# 保存为标准格式
output_dir = Path('/home/vln/imuproject/MdlVerifyV1/test_output') / unit.name
output_dir.mkdir(parents=True, exist_ok=True)

ImuDataSerializer(unit.imu_data).save(output_dir / 'imu.csv')
PosesDataSerializer(unit.gt_data).save(output_dir / 'gt.csv')

print(f'  📁 Saved to: {output_dir}')
