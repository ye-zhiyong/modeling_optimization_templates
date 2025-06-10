import pandas as pd

# 读取 XLSX 文件
xlsx_file = "dataset.xlsx"  # 输入文件
csv_file = "dataset.csv"    # 输出文件

# 读取 Excel 文件（支持多个 Sheet）
df = pd.read_excel(xlsx_file, engine="openpyxl")

# 保存为 CSV
df.to_csv(csv_file, index=False, encoding="utf-8")

print(f"转换完成！CSV 文件已保存为: {csv_file}")