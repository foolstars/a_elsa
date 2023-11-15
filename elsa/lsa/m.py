import subprocess
import time

# 定义要调用的脚本和参数
script_name = "lsa_gpu_compute.py"
input_file1 = "/home/liy/data/first_file.txt"
input_file2 = "/home/liy/data/second_file.txt"
result_file = "result"
d_value = "10"
r_value = "1"
s_value = "50"
p_value = "theo"

# python lsa_compute.py /home/liy/data/first_file.txt result -e /home/liy/data/second_file.txt -d 10 -r 1 -s 50 -p theo
# 构建命令
command = f"python {script_name} {input_file1} {result_file} -e {input_file2} -d {d_value} -r {r_value} -s {s_value} -p {p_value}"
# \
#     -m {minOccur} -p {pvalueMethod} -x {precision} -b {bootNum} -t {transFunc} -f {fillMethod} -n {normMethod} \
#     -q {qvalueMethod} -T {trendThresh} -a {approxVar} -v {progressive}"

start = time.time()
subprocess.run(command, shell=True)

print(time.time() - start)
