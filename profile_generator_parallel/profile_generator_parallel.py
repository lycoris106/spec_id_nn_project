import numpy as np
import matplotlib.pyplot as plt
import os
# from multiprocessing import Pool
import multiprocessing
import pandas as pd

# all in cgs unit
PLANCK_CONSTANT = 6.62607015E-27
SPEED_OF_LIGHT = 2.99792458E10
BOLTZMANN_CONSTANT = 1.380649E-16

h_over_c_square = PLANCK_CONSTANT / SPEED_OF_LIGHT / SPEED_OF_LIGHT
h_over_k = PLANCK_CONSTANT / BOLTZMANN_CONSTANT

###################################################################################
T_BACKGROUND = 2.732 # K
REST_FREQUENCY_ERROR_THRESHOLD = 0.1 # MHz
LOG300KI_THRESHOLD = -6 # log10 value

MAKE_PNG_PLOT = True # setting it to true will affect generation performance
N_CORE = 8
###################################################################################


# load input condition
with open("./condition.txt", 'r') as file_obj:
    tmp = file_obj.readlines()
    temperature_from = float(tmp[0].split()[2])
    temperature_to = float(tmp[1].split()[2])
    temperature_num = int(tmp[2].split()[2])
    fwhm_from = float(tmp[3].split()[2])
    fwhm_to = float(tmp[4].split()[2])
    fwhm_num = int(tmp[5].split()[2])
    filling_factor_from_log = float(tmp[6].split()[2])
    filling_factor_to_log = float(tmp[7].split()[2])
    filling_factor_num_log = int(tmp[8].split()[2])
    frequency_from = float(tmp[9].split()[2])
    frequency_to = float(tmp[10].split()[2])
    n_channel = int(tmp[11].split()[2])
    log10Ntot_from = float(tmp[12].split()[2])
    log10Ntot_to = float(tmp[13].split()[2])
    log10Ntot_num = int(tmp[14].split()[2])
    v_source = float(tmp[15].split()[2])

# load input molecular tags
manually_selected_species_tag_list = ['064502',] # c-C6H5CCH
# with open("./species_list.txt", 'r') as file_obj:
#     tmp = file_obj.readlines()
#     for ii in range(len(tmp)):
#         manually_selected_species_tag_list.append(tmp[ii][0:6].replace(" ", "0"))



obs_freq = np.linspace(frequency_from, frequency_to, n_channel, endpoint=False)
temperature_grid = np.linspace(temperature_from, temperature_to, temperature_num)
fwhm_grid = np.linspace(fwhm_from, fwhm_to, fwhm_num)
filling_factor_log_grid = np.logspace(filling_factor_from_log, filling_factor_to_log, filling_factor_num_log)
log10Ntot_grid = np.linspace(log10Ntot_from, log10Ntot_to, log10Ntot_num)



# load partition function lookup table
def load_partition_fn_table():
    with open("./raw_partition_fn_CDMS_patched.txt", 'r') as file_obj:
        eachlines = file_obj.readlines()

    def extrapolation(x, x0, y0, x1, y1):
        return (y1 - y0) / (x1 - x0) * (x - x0) + y0

    q_lookup_dict = {}
    q_temperature = [1000, 500, 300, 225, 150, 75, 37.5, 18.75, 9.375, 5.000, 2.725] # K

    for i in range(2, len(eachlines)):
        q_value = [None for j in range(len(q_temperature))]
        mol_tag = eachlines[i][0:6].replace(" ", "0")
        tmp = eachlines[i][41:-1].split()
        q_value[0] = float(tmp[0]) if "---" not in tmp[0] else None
        q_value[1] = float(tmp[1]) if "---" not in tmp[1] else None
        q_value[2] = float(tmp[2]) if "---" not in tmp[2] else None
        q_value[3] = float(tmp[3]) if "---" not in tmp[3] else None
        q_value[4] = float(tmp[4]) if "---" not in tmp[4] else None
        q_value[5] = float(tmp[5]) if "---" not in tmp[5] else None
        q_value[6] = float(tmp[6]) if "---" not in tmp[6] else None
        q_value[7] = float(tmp[7]) if "---" not in tmp[7] else None
        q_value[8] = float(tmp[8]) if "---" not in tmp[8] else None
        q_value[9] = float(tmp[9]) if "---" not in tmp[9] else None
        q_value[10] = float(tmp[10]) if "---" not in tmp[10] else None

        # estimate missing values by extrapolation
        # for high temperatures, it is extrapolation
        # for low temperatures, it is approximated as a constant
        if q_value[0] is None and q_value[1] is not None:
            q_value[0] = extrapolation(q_temperature[0], q_temperature[1], q_value[1], q_temperature[2], q_value[2])
        if q_value[0] is None and q_value[1] is None:
            q_value[0] = extrapolation(q_temperature[0], q_temperature[2], q_value[2], q_temperature[3], q_value[3])
            q_value[1] = extrapolation(q_temperature[1], q_temperature[2], q_value[2], q_temperature[3], q_value[3])
        if q_value[-2] is None and q_value[-1] is None:
            q_value[-2] = q_value[-3] * 1.0
            q_value[-1] = q_value[-3] * 1.0
        if q_value[-2] is not None and q_value[-1] is None:
            q_value[-1] = q_value[-2] * 1.0

        q_lookup_dict[mol_tag] = q_value

    return q_lookup_dict

q_lookup_dict = load_partition_fn_table()

def gup_convertor(gup_string):
     mapper = {"A": 100, "B": 110, "C": 120, "D": 130, "E": 140,
               "F": 150, "G": 160, "H": 170, "I": 180, "J": 190,
               "K": 200, "L": 210, "M": 220, "N": 230, "O": 240,
               "P": 250, "Q": 260, "R": 270, "S": 280, "T": 290,
               "U": 300, "V": 310, "W": 320, "X": 330, "Y": 340,
               "Z": 350}
     if gup_string[0].isalpha():
         return mapper[gup_string[0]] + float(gup_string[1:])
     else:
         return float(gup_string)

class Transition:
    def __init__(self, transition_string):
        # tabulated quantities
        self.mol_tag = transition_string[0:6]
        self.mol_name = transition_string[7:33].rstrip()
        self.rest_frequency = float(transition_string[34:48]) # MHz
        self.rest_frequency_error = float(transition_string[48:56]) # MHz
        self.log300KI = float(transition_string[56:64])
        self.elow = float(transition_string[66:76]) # cm^-1
        self.gup = gup_convertor(transition_string[76:79])
        self.qn = transition_string[90:-1]

        # get partition function table
        self.tabulated_t = [1000, 500, 300, 225, 150, 75, 37.5, 18.75, 9.375, 5.000, 2.725]
        self.tabulated_q = q_lookup_dict[self.mol_tag]

        # computed quantities
        self.elowK = self.elow * PLANCK_CONSTANT * SPEED_OF_LIGHT / BOLTZMANN_CONSTANT
        self.eupK = self.elowK + PLANCK_CONSTANT * self.rest_frequency * 1E6 / BOLTZMANN_CONSTANT
        self.einstein_coefficient = 2.7964E-16 * (10**self.log300KI) * self.rest_frequency**2 * 10**self.tabulated_q[self.tabulated_t.index(300)] / self.gup / (np.exp(-1.0 * self.elowK / 300.0) - np.exp(-1.0 * self.eupK / 300.0))
        self.label = self.mol_name + " --- " + self.qn

# collect possible molecular transitions within the observed frequency range
with open("./combined_database_CDMS_patched.txt", 'r') as file_obj:
    database_transitions = file_obj.readlines()


def get_candidate_transitions(mol_tag):
    candidate_transitions = []
    for ii in range(len(database_transitions)):
        if database_transitions[ii][0:6] == mol_tag and float(database_transitions[ii][34:48]) < obs_freq.max() and float(database_transitions[ii][34:48]) > obs_freq.min() and float(database_transitions[ii][48:56]) < REST_FREQUENCY_ERROR_THRESHOLD and float(database_transitions[ii][56:64]) > LOG300KI_THRESHOLD:
            candidate_transitions.append(Transition(database_transitions[ii][0:-1]))

    print(f"There are {len(candidate_transitions)} transitions for {mol_tag}...")

    return candidate_transitions


def approximated_q(tex, q_temp, q_value):
    if tex > q_temp[0]:
        return q_value[0]
    elif tex < q_temp[-1]:
        return q_value[-1]
    else:
        for i in range(0, len(q_temp)-1, 1):
            if q_temp[i] > tex and q_temp[i+1] <= tex:
                return (q_value[i+1] - q_value[i]) / (q_temp[i+1] - q_temp[i]) * (tex - q_temp[i]) + q_value[i]

def frequency_dopper_shift(nu0, v_source):
    return -1.0 * nu0 * v_source * 1E5 / SPEED_OF_LIGHT

def radiation_temperature_fn(nu, T):
    # nu in Hz
    # T in K
    return h_over_k * nu / (np.exp(h_over_k * nu / T) - 1.0)

def profile_fn(nu, nu0, sigma):
    return 1.0 / sigma / np.sqrt(2.0 * np.pi) * np.exp(-1.0 * (nu - nu0)**2 / 2.0 / sigma /sigma)

def fwhm_to_sigma(nu0, fwhm):
    return nu0 / SPEED_OF_LIGHT / np.sqrt(8.0 * np.log(2.0)) * fwhm

def tau(nu, Tex, fwhm, v_source, logNtot, candidate_transitions):
    tau_spectrum = np.zeros(len(obs_freq))
    for i in range(len(candidate_transitions)):
        tau_spectrum += SPEED_OF_LIGHT**2 / 8.0 / np.pi / (nu*1E6)**2 * 10**logNtot / 10**approximated_q(Tex, candidate_transitions[i].tabulated_t, candidate_transitions[i].tabulated_q) * candidate_transitions[i].einstein_coefficient * candidate_transitions[i].gup * np.exp(-1.0 * candidate_transitions[i].eupK / Tex) * (np.exp(h_over_k * candidate_transitions[i].rest_frequency*1E6 / Tex) - 1.0) * profile_fn(nu*1E6, (candidate_transitions[i].rest_frequency+frequency_dopper_shift(candidate_transitions[i].rest_frequency, v_source))*1E6, fwhm_to_sigma(candidate_transitions[i].rest_frequency*1E6, fwhm*1E5))

    return tau_spectrum

def synthetic_profile(nu, Tex, fwhm, beam_dilution_factor, v_source, logNtot, candidate_transitions):
    return beam_dilution_factor * (radiation_temperature_fn(nu*1E6, Tex) - radiation_temperature_fn(nu*1E6, T_BACKGROUND)) * (1.0 - np.exp(-1.0 * tau(nu, Tex, fwhm, v_source, logNtot, candidate_transitions)))



def profile_generator(shared_list, mol_tag):
    print(f"{mol_tag}...")

    # if not mol_tag in ["054518", "061503", "097501"]:
    #     return

    candidate_transitions = get_candidate_transitions(mol_tag)
    if len(candidate_transitions) == 0:
        return

    print(f"Number of model spectra: for this molecule: {temperature_num * log10Ntot_num * filling_factor_num_log * fwhm_num}")
    os.system("rm -rf ./tmp_dir/model_profile_%s"%mol_tag)
    os.system("mkdir ./tmp_dir/model_profile_%s"%mol_tag)
    model_count = 0
    for Tex in temperature_grid:
        for logNtot in log10Ntot_grid:
            for fwhm in fwhm_grid:
                for beam_dilution_factor in filling_factor_log_grid:

                    if MAKE_PNG_PLOT == True:
                        fig = plt.figure(figsize=(12, 8))
                        plt.subplot(211)
                        plt.title(f"Opacity profile - Tex:{Tex:.2f}, fwhm:{fwhm:.2f}, v_source:{v_source:.2f}, logNtot:{logNtot:.3f}")
                        plt.ylabel("tau")
                        plt.plot(obs_freq, tau(obs_freq, Tex, fwhm, v_source, logNtot, candidate_transitions))
                        plt.subplot(212)
                        plt.title(f"Model profile - beam_dilution_factor:{beam_dilution_factor:.3f}")
                        plt.xlabel("frequency (MHz)")
                        plt.ylabel("Brightness temperature (K)")
                        plt.plot(obs_freq, synthetic_profile(obs_freq, Tex, fwhm, beam_dilution_factor, v_source, logNtot, candidate_transitions))
                        plt.savefig("./tmp_dir/model_profile_%s/model_%s_%07d.png"%(mol_tag, mol_tag, model_count))
                        plt.close(fig)
                        # print(f'saved pic at:', "./tmp_dir/model_profile_%s/model_%s_%07d.png"%(mol_tag, mol_tag, model_count))

                    # np.savetxt("./tmp_dir/model_profile_%s/model_%s_%07d.tsv"%(mol_tag, mol_tag, model_count), np.c_[obs_freq, synthetic_profile(obs_freq, Tex, fwhm, beam_dilution_factor, v_source, logNtot, candidate_transitions)], fmt=["%f", "%.6E"], delimiter="\t", header="Tex:%.3e fwhm:%.3e f:%.3e v:%.3e logN:%.3e \nfrequency (GHz) brightness temperature (K)"%(Tex, fwhm, beam_dilution_factor, v_source, logNtot))
                    li = synthetic_profile(obs_freq, Tex, fwhm, beam_dilution_factor, v_source, logNtot, candidate_transitions).tolist()
                    li.append(mol_tag)
                    shared_list.append(li)
                    model_count += 1

    # print(model_count)





if __name__ == '__main__':
    # os.system(f"rm -rf ./tmp_dir")
    # os.system(f"mkdir ./tmp_dir")

    manager = multiprocessing.Manager()
    shared_list = manager.list()

    with multiprocessing.Pool(N_CORE) as p:
        # p.map(profile_generator, manually_selected_species_tag_list)
        p.starmap(profile_generator, [(shared_list, spe) for spe in manually_selected_species_tag_list])

    gathered_list = list(shared_list)
    csv_df = pd.DataFrame(gathered_list)
    csv_df.to_csv("./data_84_85_576_v0_c2000.csv", index=False)

    # dir_name = "./model_profile_400"
    # os.system(f"rm -rf {dir_name}")
    # os.system(f"mkdir {dir_name}")




    # for tag in manually_selected_species_tag_list:
    #     os.system(f"mv model_profile_%s/* {dir_name}"%tag)
    #     os.system("rm -rf model_profile_%s"%tag)

    # for d in os.listdir("./tmp_dir/"):
    #     p = os.path.join("./tmp_dir", d, "*")
    #     os.system(f"mv {p} {dir_name}")
    #     os.system(f"rm -rf {p}")
