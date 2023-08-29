import numpy as np
import matplotlib.pyplot as plt

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

FREQUENCY_FROM = 84052.8210975 # MHz
FREQUENCY_TO = 84988.80020194653 # MHz
N_CHANNEL = 1918
obs_freq = np.linspace(FREQUENCY_FROM, FREQUENCY_TO, N_CHANNEL)

T_EX = 178 # K
FILLING_FACTOR = 1
FWHM_VELO = 8 # km/s
SOURCE_VELOCITY = 0.0 # km/s

NOISE_LEVEL = 1.0 # 2.5K
###################################################################################


manual_selected_species_tag_list = ['032504', # CH3OH, vt=0-2
                                    '033502', # *C-13-H3OH, vt=0,1
                                    '039510', # c-CCC-13-H2
                                    '045512', # HC(O)NH2, v=0
                                    '054507', # H2CC-13-HCN, v=0
                                    '054510', # Propynal
                                    '060519', # a-i-C3H7OH
                                    '067503', # c-C3H5CN
                                    '070505',
                                    '076516',
                                    '076523',
                                    '077508',
                                    '080506',
                                    '083505',
                                    '083506',
                                    '090501',
                                    '113501',]

column_densities_log10 = [18.0] * 18 # CH2(OH)CHO

###################################################################################

# load partition function lookup table
def load_partition_fn_table():
    with open("./raw_partition_fn_CDMS.txt", 'r') as file_obj:
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
with open("./combined_database_CDMS.txt", 'r') as file_obj:
    database_transitions = file_obj.readlines()

candidate_transitions = []
for ii in range(len(database_transitions)):
    if len(manual_selected_species_tag_list) != 0:
        if database_transitions[ii][0:6] in manual_selected_species_tag_list and float(database_transitions[ii][34:48]) < obs_freq.max() and float(database_transitions[ii][34:48]) > obs_freq.min() and float(database_transitions[ii][48:56]) < REST_FREQUENCY_ERROR_THRESHOLD and float(database_transitions[ii][56:64]) > LOG300KI_THRESHOLD:
            candidate_transitions.append(Transition(database_transitions[ii][0:-1]))
    else:
        if float(database_transitions[ii][34:48]) < obs_freq.max() and float(database_transitions[ii][34:48]) > obs_freq.min() and float(database_transitions[ii][48:56]) < REST_FREQUENCY_ERROR_THRESHOLD and float(database_transitions[ii][56:64]) > LOG300KI_THRESHOLD:
            candidate_transitions.append(Transition(database_transitions[ii][0:-1]))

print(f"There are {len(candidate_transitions)} transitions...")

candidate_species_tag_list = []
candidate_species_name_list = []
for obj in candidate_transitions:
    if obj.mol_tag not in candidate_species_tag_list:
        candidate_species_tag_list.append(obj.mol_tag)
        candidate_species_name_list.append(obj.mol_name)

print(f"There are {len(candidate_species_tag_list)} unique species...")


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

def beam_dilution_factor(source_size, beam_size):
    return source_size**2 / (source_size**2 + beam_size**2)


def profile_fn(nu, nu0, sigma):
    return 1.0 / sigma / np.sqrt(2.0 * np.pi) * np.exp(-1.0 * (nu - nu0)**2 / 2.0 / sigma /sigma)

def fwhm_to_sigma(nu0, fwhm):
    return nu0 / SPEED_OF_LIGHT / np.sqrt(8.0 * np.log(2.0)) * fwhm

def tau(nu, Tex, fwhm, v_source, *logNtot):
    tau_spectrum = np.zeros(len(obs_freq))
    for i in range(len(candidate_transitions)):
        id = candidate_species_tag_list.index(candidate_transitions[i].mol_tag)
        tau_spectrum += SPEED_OF_LIGHT**2 / 8.0 / np.pi / (nu*1E6)**2 * 10**logNtot[id] / 10**approximated_q(Tex, candidate_transitions[i].tabulated_t, candidate_transitions[i].tabulated_q) * candidate_transitions[i].einstein_coefficient * candidate_transitions[i].gup * np.exp(-1.0 * candidate_transitions[i].eupK / Tex) * (np.exp(h_over_k * candidate_transitions[i].rest_frequency*1E6 / Tex) - 1.0) * profile_fn(nu*1E6, (candidate_transitions[i].rest_frequency+frequency_dopper_shift(candidate_transitions[i].rest_frequency, v_source))*1E6, fwhm_to_sigma(candidate_transitions[i].rest_frequency*1E6, fwhm*1E5))

    return tau_spectrum

def synthetic_profile(nu, Tex, fwhm, v_source, *logNtot):
    return FILLING_FACTOR * (radiation_temperature_fn(nu*1E6, Tex) - radiation_temperature_fn(nu*1E6, T_BACKGROUND)) * (1.0 - np.exp(-1.0 * tau(nu, Tex, fwhm, v_source, *logNtot)))


mock_profile = synthetic_profile(obs_freq, T_EX, FWHM_VELO, SOURCE_VELOCITY, *column_densities_log10)
mock_profile_noisy = mock_profile + np.random.normal(0, NOISE_LEVEL, len(obs_freq))

fig, ax = plt.subplots(2, 1)

fig.set_figwidth(12)
fig.set_figheight(8)

ax[0].plot(obs_freq, mock_profile_noisy, drawstyle="steps-mid", label='noisy')
# ax[0].plot(obs_freq, mock_profile, label='noise-free')

# ax[1].plot(obs_freq, mock_profile, label='noise-free')
# for obj in candidate_transitions:
#     ax[1].vlines(obj.rest_frequency, 0, np.max(mock_profile), color='black', alpha=0.5)
#     ax[1].text(obj.rest_frequency, 0, obj.label, rotation=90, ha='center', fontsize='small')


# ax[1].set_xlabel("Frequency (MHz)")
# ax[0].set_ylabel("Brightness temperature (K)")
# ax[1].set_ylabel("Brightness temperature (K)")
# ax[0].set_title("Synthetic spectrum")
# ax[0].legend()
plt.show()


# export transitions from the database
with open("./db_transitions.txt", 'w') as fobj:
    for obj in candidate_transitions:
        fobj.write(f"{obj.mol_tag}\t{obj.rest_frequency}\t{obj.rest_frequency_error}\t{obj.log300KI}\t{obj.eupK}\t{obj.label}\n")


# export the mock spectra
with open("./mock_spec_mal.txt", 'w') as fobj:
    fobj.write(f"#frequency\tnoise-free\tnoisy\n")
    for i in range(len(obs_freq)):
        fobj.write(f"{obs_freq[i]:.4f}\t{mock_profile[i]:.6E}\t{mock_profile_noisy[i]:.6E}\n")