"""Configuration file for common models/experiments"""

MAIN_PARAMS = { 
    'sent140': {
        'small': (10, 2, 2),
        'medium': (16, 2, 2),
        'large': (24, 2, 2)
        },
    'femnist': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 2)
        },
    'shakespeare': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (2, 1, 2)  # (20, 1, 2)
        },
    'celeba': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 2)
        },
    'synthetic': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (20, 1, 2)
        },
    'reddit': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (20, 1, 2)
        },
    'big_reddit': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (20, 1, 2)
        },
    'realworld_br': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (20, 1, 2)
        },
    'realworld_co': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (20, 1, 2)
        },
    'realworld_id': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (20, 1, 2)
        },
}
"""dict: Specifies execution parameters (tot_num_rounds, eval_every_num_rounds, clients_per_round)"""

MODEL_PARAMS = {
    'sent140.bag_dnn': (0.0003, 2), # lr, num_classes
    'sent140.stacked_lstm': (0.0003, 25, 2, 100), # lr, seq_len, num_classes, num_hidden
    'sent140.bag_log_reg': (0.0003, 2), # lr, num_classes
    'femnist.cnn': (0.0003, 62), # lr, num_classes
    'shakespeare.stacked_lstm': (0.0003, 80, 80, 256), # lr, seq_len, num_classes, num_hidden
    'celeba.cnn': (0.1, 2), # lr, num_classes
    'adult.log_reg_rw': (0.003, 2, 18), #lr, num_classes, input_dim
    'adult.log_reg': (0.003, 2, 18), #lr, num_classes, input_dim
    'compas.log_reg': (0.003, 2, 10), #lr, num_classes, input_dim
    'compas.log_reg_rw': (0.003, 2, 10), #lr, num_classes, input_dim
    'synthetic.log_reg': (0.0003, 5, 60), # lr, num_classes, input_dim
    'reddit.stacked_lstm': (0.0003, 10, 256, 2), # lr, seq_len, num_hidden, num_layers
    'reddit.topk_stacked_lstm': (0.0003, 10, 256, 2), # lr, seq_len, num_hidden, num_layers
    'big_reddit.stacked_lstm': (0.0003, 10, 256, 2), # lr, seq_len, num_hidden, num_layers
    'big_reddit.topk_stacked_lstm': (0.0003, 10, 256, 2), # lr, seq_len, num_hidden, num_layers
    'realworld_br.stacked_lstm': (0.0003, 10, 256, 2), # lr, seq_len, num_hidden, num_layers
    'realworld_br.topk_stacked_lstm': (0.0003, 10, 256, 2), # lr, seq_len, num_hidden, num_layers
    'realworld_co.stacked_lstm': (0.0003, 10, 256, 2), # lr, seq_len, num_hidden, num_layers
    'realworld_co.topk_stacked_lstm': (0.0003, 10, 256, 2), # lr, seq_len, num_hidden, num_layers
    'realworld_id.stacked_lstm': (0.0003, 10, 256, 2), # lr, seq_len, num_hidden, num_layers
    'realworld_id.topk_stacked_lstm': (0.0003, 10, 256, 2), # lr, seq_len, num_hidden, num_layers
}
"""dict: Model specific parameter specification"""

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'
