class Hyperparams:
    PAD_TOKEN = 0
    UNK_TOKEN = 1
    START_TOKEN = 2
    END_TOKEN = 3
    # cross-validation
    k_fold = 5
    # training scheme
    max_len = 20
    embed_size = 256
    encoder_banks = 5
    norm_type = "ln"
    dropout = 0.3
    lr = 0.0001
    batch_size = 128
    num_epochs = 100
    is_earlystopping = True
    # inference
    candidate_size = 5
    beam_width = 5
    # data path
    modelname = 'first'
    logdir = "./logdir"
