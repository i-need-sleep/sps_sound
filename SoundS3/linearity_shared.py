from os import path

USING_METRIC = 'R2'
METRIC_DISPLAY = dict(
    ylabel='$R^2$', 
    rotation=0,
    labelpad=15,
)

# USING_METRIC = 'diffStd'
# METRIC_DISPLAY = dict(
#     ylabel='Std of Diff', 
# )

# USING_METRIC = 'linearProjectionMSE'
# METRIC_DISPLAY = dict(
#     ylabel='Linear Projection MSE', 
# )

# USING_METRIC = 'linearProjectionStdErr'
# METRIC_DISPLAY = dict(
#     ylabel='Linear Projection Std Error', 
# )

SPICE = 'SPICE'

# EXP_GROUPS = [
#     # display name, path name
#     ('VAE aug $\\times 4$, lock $z_\\mathrm{timbre}$', 'vae_symm_4_repeat'), 
#     ('VAE aug $\\times 4$, lock $z_\\mathrm{timbre}$ 10D', 'vae_symm_4_repeat_timbre10d'), 
#     (' AE aug $\\times 4$, lock $z_\\mathrm{timbre}$', 'ae_symm_4_repeat'), 
#     ('VAE aug $\\times 0$, lock $z_\\mathrm{timbre}$', 'vae_symm_0_repeat'), 
#     ('VAE aug $\\times 4$, RNN  $z_\\mathrm{timbre}$', 'vae_symm_4_no_repeat'), 
#     ('SPICE', SPICE), 
# ]

EXP_GROUPS = [
    # display name, path name
    # ('SPS (Ours)', 'vae_symm_4_repeat'), 
    # ('Ours w/o Symmetry', 'vae_symm_0_repeat'), 
    # ('$\\beta$-VAE (Baseline)', 'beta_vae'), 
    # ('SPICE (Baseline)', SPICE), 
    ('Nottingham: SPS 1dim', 'nottingham_eighth_accOnly_5000_2_512_easier_gru_1dim_checkpoint_9000'),
    ('Nottingham: SPS 1dim, noSymm', 'nottingham_eighth_accOnly_5000_2_512_easier_gru_noSymm_1dim_checkpoint_9000'),
    ('Scale: SPS 1dim', 'scale_1dim_checkpoint_9000'),
    ('Scale: SPS 1dim, noSymm', 'scale_no_Symm_1dim_checkpoint_9000')
]

TASKS = [
    # path name, display name, x, y, plot style
    (
        'encode', 'Embedding', 
        ('pitch', 'Pitch'), 
        ('z_pitch', '$z_\\mathrm{pitch}$'),
        dict(
            linestyle='none', 
            marker='.', 
            markersize=3, 
        ), 
    ), 
    (
        'decode', 'Synthesis', 
        ('z_pitch', '$z_\\mathrm{pitch}$'), 
        ('yin_pitch', 'Detected Pitch'),
        dict(
            linestyle='none', 
            marker='.', 
            markersize=3, 
        ), 
    ), 
]

DATA_SETS = [
    # path name, display name
    # ('train_set', 'Train'), 
    ('test_set', 'Test'), 
]

RESULT_PATH = './linearityEvalResults/%s_%s_%s/'
SPICE_PATH = './SPICE_results/result_short.txt'

COMMON_INSTRUMENTS = ['Accordion',
    'Piano', 'Accordion', 'Clarinet', 'Electric Piano', 
    'Flute', 'Guitar', 'Saxophone', 'Trumpet', 'Violin', 
    # 'Church Bells', 
]

def readXYFromDisk(
    is_SPICE, 
    result_path, 
    x_path, y_path, 
):
    data = {}
    if is_SPICE:
        if 'decode' in result_path or 'train_set' in result_path:
            raise NoSuchSpice
        with open(SPICE_PATH, 'r') as f:
            for line in f:
                line: str = line.strip()
                line = line.split('single_note_GU/')[1]
                filename, z_pitch = line.split('.wav ')
                z_pitch = float(z_pitch)
                instrument_name, pitch = filename.split('-')
                pitch = int(pitch)
                if instrument_name in COMMON_INSTRUMENTS:
                    if instrument_name not in data:
                        data[instrument_name] = ([], [])
                    X, Y = data[instrument_name]
                    X.append(pitch)
                    Y.append(z_pitch)
    else:
        for instrument_name in COMMON_INSTRUMENTS:
            if instrument_name not in COMMON_INSTRUMENTS:
                continue
            X = []
            Y = []
            def f(output: list, s: str):
                with open(path.join(
                    result_path.replace('test_set_',''), instrument_name + f'_{s}.txt'
                ), 'r') as f:
                    for line in f:
                        output.append(float(line.strip()))
            f(X, x_path)
            f(Y, y_path)
            data[instrument_name] = (X, Y)
    return data

class NoSuchSpice(Exception): pass
