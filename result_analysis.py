import pickle
from tabulate import tabulate
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

SPEED_OF_LIGHT = 299792.458

def plot_all(input_file, radial_v, istxt, freq_unit, class_dict, trans_dict, db_data, y_unit="K", bmaj=3.446816936134E-04, bmin=2.289818422717E-04):
    if istxt:
        spec_df = pd.read_csv(f"Spectra/{input_file}.txt", delimiter='\t', skiprows=1, usecols=[0, 2], names=['freq', 'value'])
    else:
        spec_df = pd.read_csv(f"Spectra/{input_file}.tsv", sep='\t', comment='#', header=None, names=['freq', 'value'])

    if freq_unit=='MHz':
        spec_df[['freq']] = spec_df[['freq']] / 1000
    # plt.figure(figsize=(16,9))
    # plt.plot(spec_df['freq'], spec_df['value'], drawstyle='steps-mid', color="orange")
    # plt.show()
    spec_df[['freq']] = spec_df[['freq']] / (1 - radial_v / SPEED_OF_LIGHT)

    if y_unit != "K":
        print(f"Converting unit from {y_unit} to K...")
        # Convert the frequency values to the assumed unit
        if y_unit == "Jy/beam":
            spec_df[['value']] = list(map(lambda x, y: y*1.22*1e6 / (x*x*bmaj*bmin), spec_df[['freq']].to_numpy(), spec_df[['value']].to_numpy()))


    db_data['freq'] = db_data['freq']/1000.0
    # db_data = db_data[(db_data['integ_intens'] > integ_intens_thres) & (db_data['uncertain'] < uncertain_thres)]


    plt.figure(figsize=(16,9))
    # fig, ax = plt.subplots()
    plt.plot(spec_df['freq'], spec_df['value'], drawstyle='steps-mid', color="orange")

    all_id = list(class_dict.keys())

    for id in all_id:
        if class_dict[id]["correct"]:
            for match_tran in mole_hit_trans_dict[id]["matched_in"]:
                r = db_data[db_data['trans_id'] == match_tran]
                plt.vlines(r['freq'], 0, np.max(spec_df['value']), color='g', alpha=0.1)
                plt.text(r['freq'], 0, r["name"].values[0], rotation=90)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for post-processing and showing the result of the CNN model')
    parser.add_argument('--input_file', type=str, help='name of original input spectrum')
    parser.add_argument('--plot_all', action='store_true', help="whether to plot the analysis result")
    parser.add_argument('--rad_v', type=float, help='the radial velocity, required if plot_all == true')
    parser.add_argument('--istxt', action='store_true', help='is the spectrum a text file, required if plot_all == true')
    parser.add_argument('--freq_unit', choices=["GHz", "MHz"], help="frequency unit of the original spectrum, required if plot_all == true: 'GHz' or 'MHz'")
    parser.add_argument('--y_unit', choices=["K", "Jy/beam"], help="unit of the y/intensity value for the original spectrum, required if plot_all == true: 'K' or 'Jy/beam'")
    parser.add_argument('--bmaj', type=float, default=3.446816936134E-04, help='bmaj, should be provided if y_unit == Jy/beam')
    parser.add_argument('--bmin', type=float, default=2.289818422717E-04, help='bmin, should be provided if y_unit == Jy/beam')


    args = parser.parse_args()

    BASE_PATH = './'

    with open(os.path.join(BASE_PATH, f"temp_files/{args.input_file}_result.pkl"), 'rb') as f:
        class_dict = pickle.load(f)

    with open(os.path.join(BASE_PATH, f"temp_files/{args.input_file}_mole_hit_trans.pkl"), 'rb') as f:
        mole_hit_trans_dict = pickle.load(f)

    # print(mole_hit_trans_dict)

    db_path = os.path.join(BASE_PATH, "data/database_csv_patched.csv")
    db_data = pd.read_csv(db_path)

    # correct = [61505, 94503, 83506, 83503, 46524, 39510, 64516, 70505, 46518, 44504, 62503, 32504, 45512, 93503, 76501, 47518, 85501, 61502]
    # incorrect = [94501, 34504, 83502, 80503, 43517, 56510, 55502, 60505, 26502, 108501, 58518, 69509, 69510, 62504, 76521, 76522, 61518, 62524, 69511, 83505, 83507, 74519, 80506, 76518, 76513, 76515, 57507, 60519, 47511, 76516, 53515, 68506, 79501, 72505, 75511, 74514, 45513, 78505, 67502, 74515, 56511, 76519, 60999, 61517, 73503, 69513, 58517, 69506, 38502, 76520, 45515, 75512, 55515, 33502, 60517, 106501, 45516, 67503, 80505, 74503, 49510, 57508, 57512, 58505, 92502, 13502, 54508, 73501, 66506, 60524, 72504, 96501, 99501, 58519, 65514, 61515, 99502, 67504, 44506, 55505, 76523, 56504, 99503, 54507, 65502, 89502, 58508, 74516, 84503, 103501, 70504]

    detect_dict = dict()
    with open("data/detection.txt", 'r') as file_obj:
        tmp = file_obj.readlines()
        for ii in range(len(tmp)):
            detect_dict[int(tmp[ii][0:6])] = tmp[ii][7:].strip()
    all_id = list(class_dict.keys())
    all_id.sort()
    result_dict = {"id":[], "result":[], "mole_name":[], "matched: y/n/out":[], "pred-1":[], "pred-2":[], "inISM":[]}
    for id in all_id:
        result_dict["id"].append(id)
        result_dict["mole_name"].append(db_data[db_data["id"] == id].head(1)["name"].values[0])
        result_dict["result"].append("\x1b[32mCorrect\x1b[0m" if class_dict[id]["correct"] else "\x1b[31mIncorrect\x1b[0m")
        m1 = len(mole_hit_trans_dict[id]["matched_in"])
        result_dict["matched: y/n/out"].append(str(m1) if m1 > 1 else "\x1b[31m"+str(m1)+"\x1b[0m")
        result_dict["matched: y/n/out"][-1] =  result_dict["matched: y/n/out"][-1] + "/" + str(len(mole_hit_trans_dict[id]["not_matched_in"]))
        result_dict["matched: y/n/out"][-1] =  result_dict["matched: y/n/out"][-1] + "/" + str(len(mole_hit_trans_dict[id]["matched_out"]))
        result_dict["pred-1"].append("\x1b[34m"+str(class_dict[id]["1-pred"])+"\x1b[0m" if m1 < 2 else class_dict[id]["1-pred"])
        result_dict["pred-2"].append("\x1b[34m"+str(class_dict[id]["2-pred"])+"\x1b[0m" if m1 < 2 else class_dict[id]["2-pred"])
        result_dict["inISM"].append("" if not id in detect_dict else detect_dict[id])



    print(tabulate(result_dict, headers='keys', tablefmt='psql'))

    if args.plot_all:
        plot_all(input_file=args.input_file, radial_v=args.rad_v, istxt=args.istxt, freq_unit=args.freq_unit, class_dict=class_dict, trans_dict=mole_hit_trans_dict, db_data=db_data, y_unit=args.y_unit, bmaj=args.bmaj, bmin=args.bmin)
