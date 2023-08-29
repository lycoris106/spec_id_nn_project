import pandas as pd

csv_path = "./profile_generator_parallel/data_84_85_576_v0_c2000.csv"
df = pd.read_csv(csv_path)

filt_df = df[df.iloc[:, :-1].sum(axis=1) != 0]

filt_df.to_csv("./data_84_85_576_v0_c2000_fil.csv", index=False)