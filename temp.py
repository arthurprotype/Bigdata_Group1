def CNN(input_shape=(256,1),inception=True, res=True, strided=True, maxpool=False, avgpool=False, batchnorm=True):
    config = {
        #             'num_steps' : X_train.shape[1],
        #             'sensors' : X_train.shape[2],
        # LSTM
        'state_size': 32,

        # CNN
        'filters': 32,
        'strides': 2,

        # Output
        'output_size': 3,

        # Activations
        'c_act': 'relu',
        'r_act': 'hard_sigmoid',
        'rk_act': 'tanh',
        'batch_size': 512,
        'learning_rate': 0.0012,
        'epochs': 200,
        'reg': 0.001,
        'rec_drop': 0.32,
        'drop': 0.5,
        'cnn_drop': 0.6,
    }
    i = 0
    pad = 'same'
    padp = 'same'

    c_act = config['c_act']
    r_act = config['r_act']
    rk_act = config['rk_act']

    r = regularizers.l2(config['reg'])

    input = Input(input_shape)
    c = input
    stride_size = config['strides'] if strided else 1

    if inception:
        c0 = Conv1D(config['filters'], kernel_size=4, strides=stride_size, padding=pad, activation=c_act)(c)
        c1 = Conv1D(config['filters'], kernel_size=8, strides=stride_size, padding=pad, activation=c_act)(c)
        c2 = Conv1D(config['filters'], kernel_size=32, strides=stride_size, padding=pad, activation=c_act)(c)

        c = concatenate([c0, c1, c2])

        if maxpool:
            c = MaxPooling1D(2, padding=padp)(c)
        elif avgpool:
            c = GlobalAveragePooling1D(2, padding=padp)(c)
        if batchnorm:
            c = BatchNormalization()(c)
        c = SpatialDropout1D(config['cnn_drop'])(c)

        c0 = Conv1D(config['filters'], kernel_size=4, strides=stride_size, padding=pad, activation=c_act)(c)
        c1 = Conv1D(config['filters'], kernel_size=8, strides=stride_size, padding=pad, activation=c_act)(c)
        c2 = Conv1D(config['filters'], kernel_size=32, strides=stride_size, padding=pad, activation=c_act)(c)

        c = concatenate([c0, c1, c2])
        if maxpool:
            c = MaxPooling1D(2, padding=padp)(c)
        elif avgpool:
            c = GlobalAveragePooling1D(2, padding=padp)(c)
        if batchnorm:
            c = BatchNormalization()(c)
        c = SpatialDropout1D(config['cnn_drop'])(c)

        c0 = Conv1D(config['filters'], kernel_size=4, strides=stride_size, padding=pad, activation=c_act)(c)
        c1 = Conv1D(config['filters'], kernel_size=8, strides=stride_size, padding=pad, activation=c_act)(c)
        c2 = Conv1D(config['filters'], kernel_size=32, strides=stride_size, padding=pad, activation=c_act)(c)

        c = concatenate([c0, c1, c2])
        if maxpool:
            c = MaxPooling1D(2, padding=padp)(c)
        elif avgpool:
            c = GlobalAveragePooling1D(2, padding=padp)(c)
        if batchnorm:
            c = BatchNormalization()(c)
        c = SpatialDropout1D(config['cnn_drop'])(c)
    if res:  # Residual RNN
        g1 = GRU(config['state_size'],
                 return_sequences=True,
                 activation=rk_act,
                 recurrent_activation=r_act,
                 dropout=config['rec_drop'],
                 recurrent_dropout=config['rec_drop'],
                 recurrent_regularizer=r,
                 kernel_regularizer=r)(c)
        g2 = GRU(config['state_size'],
                 return_sequences=True,
                 activation=rk_act,
                 recurrent_activation=r_act,
                 dropout=config['rec_drop'],
                 recurrent_dropout=config['rec_drop'],
                 recurrent_regularizer=r,
                 kernel_regularizer=r)(g1)
        g_concat1 = concatenate([g1, g2])

        g3 = GRU(config['state_size'],
                 return_sequences=True,
                 activation=rk_act,
                 recurrent_activation=r_act,
                 dropout=config['rec_drop'],
                 recurrent_dropout=config['rec_drop'],
                 recurrent_regularizer=r,
                 kernel_regularizer=r)(g_concat1)
        g_concat2 = concatenate([g1, g2, g3])

        g = GRU(config['state_size'],
                return_sequences=False,
                activation=rk_act,
                recurrent_activation=r_act,
                dropout=config['rec_drop'],
                recurrent_dropout=config['rec_drop'],
                recurrent_regularizer=r,
                kernel_regularizer=r)(g_concat2)
    d = Dense(config['output_size'])(g)
    out = Softmax()(d)

    model = Model(input, out)
    print("{} initialized.".format(model.name))

    adam = optimizers.Adam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=0.00000001, decay=0.0, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model