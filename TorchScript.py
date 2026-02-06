import torch


def my_function(x):
    for i in range(10):
        if i == 2:
            x = x + i
            return x
    return x + 1


# 编译函数
scripted_function = torch.jit.script(my_function)

# 使用编译后的函数
x = torch.Tensor([1])
print(f"x = {x}")
result = scripted_function(x)
print(result)


scripted_function.save("my_function.pt", {})


loaded = torch.jit.load("my_function.pt")
result = loaded(x)
print(result)
