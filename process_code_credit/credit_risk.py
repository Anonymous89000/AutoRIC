import csv

headers = ['status', 'duration', 'credit_history',
               'purpose', 'amount',
               'savings', 'employment_duration',
               'installment_rate', 'personal_status_sex', 'other_debtors',
               'present_residence', 'property',
               'age', 'other_installment_plans',
               'housing', 'number_credits',
               'job', 'people_liable',
               'telephone', 'foreign_worker',
               'credit_risk'] 

# 打开原始数据文件和输出的 CSV 文件
with open('data/initial/german.data', 'r', encoding='utf-8') as input_file, open('data/initial/german.csv', 'w', encoding='utf-8', newline='') as output_file:
    # 创建一个 CSV writer 对象
    writer = csv.writer(output_file)

    # 写入表头
    writer.writerow(headers)

    # 读取输入文件的每一行
    for line in input_file:
        # 拆分每一行的字段
        fields = line.strip().split()

        # 将字段写入 CSV 文件
        writer.writerow(fields)
