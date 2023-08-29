import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
# from tqdm import tqdm
# from scipy import constants
import re
from tabulate import tabulate
import argparse
import pickle


SPEED_OF_LIGHT = 299792.458
MODEL_FREQ_MAX = 85.0 # excluded
MODEL_FREQ_MIN = 84.0
MODEL_N_CHANNELS = 2000
MODEL_CHANNEL_WIDTH=(MODEL_FREQ_MAX - MODEL_FREQ_MIN) / MODEL_N_CHANNELS

def gaussian_add(mole_dict, d):
    # def freq2chan(num):
    #     return (num - freq_min) * (n_channel - 1) / (freq_max - freq_min)
    def chan2freq(chan):
        return MODEL_FREQ_MIN + chan * MODEL_CHANNEL_WIDTH

    for i in range(len(mole_dict[d['id']])):
        if abs(chan2freq(i)-d['mean']) < d['fwhm']*5:
            val = d['amplitude'] * np.exp(-4. * np.log(2) * (chan2freq(i)-d['mean'])**2 / d['fwhm']**2)
            mole_dict[d['id']][i] += val
        # if d['id'] == 32504:
        #     print(chan2freq(i), d['mean'], d['fwhm'], d['amplitude'])


def match_database(input_file, integ_intens_thres=-6, uncertain_thres=0.1, print_match=False, do_plot=False, freq_unit='GHz', istxt=False, y_unit="K", bmaj=3.446816936134E-04, bmin=2.289818422717E-04):
    """
    Input: A tsv file of gaussians that fit the original spectrum, radial velocity (in )
    Output: A csv file of gaussians that match transitions in the CDMS database (molecule_id, mean, amplitude, fwhm)
    """

    database_file = r'data/database_csv_patched.csv'
    database_df = pd.read_csv(database_file)

    with open(f"temp_files/{input_file}_gauss_param_v0.csv", 'r') as file:
        for line in file:
            if line.startswith('#'):
                # Extract parameters
                pattern = r"#\s*freq_min:(\d+\.*\d*),\s*freq_max:(\d+\.*\d*),\s*n_channels:(\d+),\s*radial_v:(\d+\.*\d*),\s*fwhm_guess:(\d+\.*\d*)"

                match = re.match(pattern, line)
                if match:
                    data = match.groups()
                    freq_min = float(data[0])
                    freq_max = float(data[1])
                    n_channel = int(data[2])
                    radial_v = float(data[3])
                    fwhm_guess = float(data[4])
                    print(f"freq_min:{freq_min}, freq_max:{freq_max}, n_channel:{n_channel}, radial_v:{radial_v}, fwhm_guess:{fwhm_guess}")
                else:
                    print("Failed to extract parameters.")
                    exit()

                break # read only one line of comments


    prop_df = pd.read_csv(f"temp_files/{input_file}_gauss_param_v0.csv", sep='\t', comment='#')

    # adjust for Doppler shift and unit
    database_df['freq'] = database_df['freq']/1000.0
    # database_df['freq_shift'] = database_df['freq'] - (radial_velocity * database_df['freq']) / SPEED_OF_LIGHT
    # database_df['freq_shift'] = database_df['freq']
    # prop_df['mean'] = prop_df['mean']

    # filter out transitions in database whose (1) intensity is low (2) uncertainty is large
    database_df = database_df[(database_df['integ_intens'] > integ_intens_thres) & (database_df['uncertain'] < uncertain_thres)]

    # channel_width = (freq_max - freq_min) / (n_channel - 1)
    trans_gauss_list = []

    # delta = channel_width*prop_df.loc[0, 'fwhm']*0.7
    delta = MODEL_CHANNEL_WIDTH*fwhm_guess*0.7
    peak_count = 0
    count = 0
    mole_set = set()
    # trans_hit = pd.DataFrame([])

    if do_plot:
        if istxt:
            spec_df = pd.read_csv(f"Spectra/{input_file}.txt", delimiter='\t', skiprows=1, usecols=[0, 2], names=['freq', 'value'])
        else:
            spec_df = pd.read_csv(f"Spectra/{input_file}.tsv", sep='\t', comment='#', header=None, names=['freq', 'value'])

        if freq_unit=='MHz':
            spec_df[['freq']] = spec_df[['freq']] / 1000

        spec_df[['freq']] = spec_df[['freq']] / (1 - radial_v / SPEED_OF_LIGHT)

        if y_unit != "K":
            print(f"Converting unit from {y_unit} to K...")
            # Convert the frequency values to the assumed unit
            if y_unit == "Jy/beam":
                spec_df[['value']] = list(map(lambda x, y: y*1.22*1e6 / (x*x*bmaj*bmin), spec_df[['freq']].to_numpy(), spec_df[['value']].to_numpy()))


        fig, ax = plt.subplots()
        ax.plot(spec_df['freq'], spec_df['value'], drawstyle='steps-mid', color="orange")


    highest = -1
    high_peak = None
    peak_match_count = 0
    mole_dict = defaultdict(lambda: np.array([0.0] * MODEL_N_CHANNELS))
    mole_hit_trans_dict = defaultdict(lambda: {'matched_in': set(), 'matched_out': set(), 'not_matched_in': set()})
    for index, props in prop_df.iterrows():
        # print(type(props['fwhm']), type(props['amplitude']), type(props['mean']))
        candid_database = database_df[(database_df['freq'] > props['mean']-delta) & (database_df['freq'] < props['mean']+delta)]
        hit_set = set(candid_database['id'])
        for h in hit_set:
            d = {"id": h, "mean": props['mean'], "amplitude": props['amplitude'], "fwhm": fwhm_guess*MODEL_CHANNEL_WIDTH}
            trans_gauss_list.append(d)
            gaussian_add(mole_dict, d)
        mole_set.update(hit_set)
        for idx, row in candid_database.iterrows():
            # check if a transition is matched with more than one peak
            # print(row)
            if row['trans_id'] in mole_hit_trans_dict[row['id']]:
                print(f"the transition {row['trans_id']} is matched again!")
            if row['freq'] > MODEL_FREQ_MIN and row['freq'] < MODEL_FREQ_MAX:
                mole_hit_trans_dict[row['id']]['matched_in'].add(row['trans_id'])
            else:
                mole_hit_trans_dict[row['id']]['matched_out'].add(row['trans_id'])
        # trans_hit = pd.concat([trans_hit, candid_database]).drop_duplicates()
        count += candid_database.shape[0]
        if candid_database.shape[0] > 0:
            peak_match_count += 1

        if do_plot:
            font_dict = {'family':'cursive','color':'#2596be','size':12}
            ax.axvline(x=props['mean'], color='#be4d25', linestyle='-', lw = 0.8, label='peak frequency')
            ax.text(props['mean'], max(spec_df['value'])/2, str(candid_database.shape[0]), color="red", rotation=90)

            for i, (index, hit_tran) in enumerate(candid_database.iterrows()):
                ax.axvline(x=hit_tran['freq'], color='#7cc0d8', linestyle='-', lw = 0.8, label='matching transition')
                ax.text(hit_tran['freq'], 0, str(hit_tran['id'])+' '+hit_tran['name'].strip()+' '+str(hit_tran['freq'])+' '+hit_tran['q'].strip(), rotation=90)


        if print_match:
            print(f"peak at {props['mean']}:")
            print(tabulate(candid_database[["id", "name", "freq", "q_code", "q"]], headers='keys', tablefmt='psql'))
            if (props['amplitude'] > highest):
                highest = props['amplitude']
                high_peak = candid_database
        peak_count += 1

    print(f'delta: {delta}')
    print(f'peak_count: {peak_count}')
    print(f'peak hit count: {peak_match_count}')
    print(f'total hit count: {count}')
    print(f'id count: {len(mole_set)}')

    # print(database_df[database_df['id'] == 32504])
    # if print_match:
    #     print(tabulate(high_peak[["id", "name", "freq_shift", "freq", "q_code", "q"]], headers='keys', tablefmt='psql'))

    if do_plot:

        plt.show()


    trans_gauss_df = pd.DataFrame(trans_gauss_list)
    trans_gauss_df = trans_gauss_df.sort_values(by=['id'])
    trans_gauss_df.to_csv(f'temp_files/{input_file}_trans_gauss_prop.csv', index=False)

    out_df = pd.DataFrame.from_dict(mole_dict)

    # Transpose the DataFrame so that keys become the last column
    out_df = out_df.T.reset_index()
    name_col = out_df.pop(out_df.columns[0])
    out_df[name_col.name] = name_col
    out_df.columns = range(len(out_df.columns))
    print(f'Writing to {input_file}_in.csv...')
    out_df.to_csv(f'temp_files/{input_file}_in.csv', index=False)

    # also write the result of matched transions into a pkl
    trans_out_name = f'temp_files/{input_file}_mole_hit_trans.pkl'

    for m in mole_set:
        mole_db = database_df[(database_df['id'] == m)]
        for idx, row in mole_db.iterrows():
            if row['freq'] > MODEL_FREQ_MIN and row['freq'] < MODEL_FREQ_MAX:
                if not row['trans_id'] in mole_hit_trans_dict[m]['matched_in']:
                    mole_hit_trans_dict[m]['not_matched_in'].add(row['trans_id'])
    # for key in sorted(mole_hit_trans_dict.keys()):
    #     values = mole_hit_trans_dict[key]
        # print(f"{key}: {', '.join(str(val) for val in values)}")
    with open(trans_out_name, 'wb') as f:
        print(f'Writing to {trans_out_name}...')
        # print(dict(mole_hit_trans_dict))
        pickle.dump(dict(mole_hit_trans_dict), f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script for matching the peaks with the database and outputting the spectrum for each candidate molecule')
    parser.add_argument('--input_file', type=str, help='input file of identified peaks')
    # parser.add_argument('--spec_file', type=str, help='spectrum file for debug mode')
    # parser.add_argument('--rad_v', type=float, help='radial velocity relative to the observer')
    # parser.add_argument('--fwhm_guess', type=float, help="fixed guess of fwhm")
    parser.add_argument('--do_plot', action='store_true', help='whether to plot the matching result')
    parser.add_argument('--print_match', action='store_true', help='whether to print the matching result')
    parser.add_argument('--freq_unit', choices=["GHz", "MHz"], help="unit for frequency in the original spectrum: 'GHz' or 'MHz'")
    parser.add_argument('--y_unit', choices=["K", "Jy/beam"], help="unit for the y/intensity value of the original spectrum: 'K' or 'Jy/beam'")
    parser.add_argument('--bmaj', type=float, default=3.446816936134E-04, help='bmaj, should be provided if y_unit == Jy/beam')
    parser.add_argument('--bmin', type=float, default=2.289818422717E-04, help='bmin, should be provided if y_unit == Jy/beam')

    parser.add_argument('--istxt', action='store_true', help='is the original file in txt format instead of csv/tsv')

    args = parser.parse_args()

    # match_database(
    #     args.input_file
    #     , radial_velocity=98.0
    #     , print_match=True
    #     , debug=args.debug
    # )

    # filename = "member.uid___A001_X1284_X6c1._G31.41p0.31__sci.spw25.cube.I.pbcor_subimage.fits-Z-profile-Region_1-Statistic_Mean-Cooridnate_Current-2023-07-11-10-21-17"
    # filename = "mock_spec"
    match_database(
        input_file=args.input_file
        , do_plot=args.do_plot
        , print_match=args.print_match
        , freq_unit=args.freq_unit
        , istxt=args.istxt
        , y_unit=args.y_unit
        , bmaj=args.bmaj
        , bmin=args.bmin
    )
