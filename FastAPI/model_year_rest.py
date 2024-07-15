import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt
import os


def get_predictions_rest(device_name, device_age, start_point, end_point): 
    # parameters
    phase_type = '3-phase'          ## Phase
    appliance_name = '3ph_fridge_1' ## Select Appliance Name Here
    model_id = "NILM_Ensemble"


    skiping_factor = 4              ## Skiping factor for the data (1 = no skiping)
    power_on_z_score = 0.25         ## ON/OFF threshold float between 0 and 1
    sample_size = 42000             ## Samples per batch
    depth = 16                      ## Change the number of dimmension the wavenet process  (Won't change Window Size)
    n_layers = 6                    ## Number of wavenet layers (Will Change Window Size)
    kernel_size = 3                 ## CNN Kernel Size   (Will change window size)
    dilation_size = None            ## CNN dilation size
    initial_lr = 0.01               ## Initial learning rate of optimizer

    # training parameters
    EPOCHS = 200                    ## Number of epochs to train for


    current_directory = os.getcwd()
    print("Current Directory:", current_directory)


    # import data
    csv_path = os.path.join(current_directory,"data",f"{device_name}.csv")
    df = pd.read_csv(csv_path)

    df_aggregate = df[df.columns[:3]]

    df_aged = df[df.columns[3:]]


    class wavenet_Unit(tf.keras.layers.Layer):
        def __init__(self, out_channels, kernel_size, dilation_rate, causal=True, residual = True, **kwargs):
            super(wavenet_Unit, self).__init__(**kwargs)
            self.causal = causal
            self.dilation_rate = dilation_rate
            self.kernel_size = kernel_size
            self.out_channels = out_channels
            self.conv1 = tf.keras.layers.Conv1D(out_channels*2, kernel_size, padding='causal' if causal else 'same', dilation_rate=dilation_rate, activation='tanh')
            self.conv2 = tf.keras.layers.Conv1D(out_channels*2, kernel_size, padding='causal' if causal else 'same', dilation_rate=dilation_rate, activation='sigmoid')
            self.out = tf.keras.layers.Conv1D(out_channels, 1, padding='causal' if causal else 'same')
            self.norm = tf.keras.layers.BatchNormalization()
            self.residual = residual

        def call(self, inputs, training=None, mask=None):
            ## Normalized the input
            ## Passed separetely to tanh and sigmoid parallely.
            norm_inputs = self.norm(inputs)
            tanh = self.conv1(norm_inputs)
            sigmoid = self.conv2(norm_inputs)
            ## Multiply both features
            x = keras.layers.Multiply()([tanh, sigmoid])

            ##Pass it through a conv layer
            x = self.out(x)
            without_res = x
            if self.residual:
                x = keras.layers.Add()([x,inputs])
            return x, without_res
            # return x

        def get_config(self):
            config = super(wavenet_Unit, self).get_config()
            config.update({
                'causal': self.causal,
                'dilation_rate': self.dilation_rate,
                'kernel_size': self.kernel_size,
                'out_channels': self.out_channels})
            return config

    class wavenet(tf.keras.Model):
        def __init__(self, n_layers = 10, out_channels = 256, kernel_size = 2, dialtion_base = 2, causal = True, base_activation = 'softmax', middle_layers_activation = 'softplus', bias_initializer = None, **kwargs) -> None:
            super(wavenet, self).__init__(**kwargs)
            self.wavenet_layers = []
            for i in range(n_layers):
                self.wavenet_layers.append(wavenet_Unit(out_channels = out_channels, kernel_size = kernel_size, dilation_rate = dialtion_base**i, causal = causal))
            self.bottom = keras.Sequential([
            keras.layers.Activation(middle_layers_activation),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(out_channels//2, activation= middle_layers_activation),
            keras.layers.Dense(out_channels//4, activation= middle_layers_activation),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1, activation=base_activation, bias_initializer=bias_initializer)
            ])

        def call(self, inps):
            skip_value = 0
            for layer in self.wavenet_layers:
                inps, single_skip_value = layer(inps)
                skip_value += single_skip_value
            return self.bottom(skip_value)


    def make_main_model(x_train, y_train, depth = 16, n_layers = 3, kernel_size = 3, dilation_size = None, middle_layers_activation = 'softplus', power_on_z_score = 0.25):
        if dilation_size is None:
            dilation_size = kernel_size

        train_output = y_train
        threshold = max(train_output.mean()-train_output.std()*power_on_z_score,0)
        off_bool = np.sum(train_output<=threshold)/train_output.shape[1]
        mean = train_output.mean()-x_train.mean()
        bias = np.log([(1-off_bool)/off_bool])
        print('mean:',mean)
        print('Threshold:', threshold)
        print('Bias:',bias)
        print('Off-bool:',off_bool)


        aggregate_input = tf.keras.Input((None,3),name='aggregate_input')

            ## Regression Model is a wevenet model but without Causal Padding nor SE; The Last layer activations is relut to limit the ouput to positive.
        regression_model = wavenet(
                n_layers = n_layers,
                kernel_size = kernel_size,
                dialtion_base = dilation_size,
                out_channels = depth,
                causal = False,
                base_activation = 'relu',
                middle_layers_activation = middle_layers_activation,
                bias_initializer = keras.initializers.Constant(mean),
                name = 'regression_model')

            ## Classification Model is a wavenet model but without Causal Padding nor SE; The last layer activation is sigmoid to get binary probabilistic output.
        classification_model = wavenet(
                n_layers = n_layers,
                kernel_size = kernel_size,
                dialtion_base = dilation_size,
                out_channels = depth,
                causal = False,
                base_activation = 'sigmoid',
                middle_layers_activation = middle_layers_activation,
                bias_initializer = keras.initializers.Constant(bias),
                name = 'classification_model')

        ONOFF = classification_model(keras.layers.Dense(depth)(aggregate_input))


            ## The Classification output is concatenated with initial aggregate input to be feeded to regression model
        concatenate_list = [ONOFF,aggregate_input]
        power_input = keras.layers.Concatenate()(concatenate_list)

            ## Small model to increase the dimmension of the regression input

        power_input = keras.Sequential([
                                            keras.layers.Conv1D(depth//2, 2, padding='same'),
                                            keras.layers.Conv1D(depth, 2, padding='same')
            ], name='Depth_Increase')(power_input)

        power = regression_model(power_input)

        optimizer = tf.keras.optimizers.Nadam(initial_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
            # optimizer = tf.keras.optimizers.SGD(0.1)
        main_model = tf.keras.Model(inputs=aggregate_input,outputs=[power,ONOFF],name = 'NILM_Model')

        main_model.compile(
                optimizer,
                ['mse',keras.losses.BinaryCrossentropy()],
                metrics={
                    'classification_model':'acc',
                })
        return main_model

    def load_model_for_appliance(depth, n_layers, kernel_size, dilation_size, middle_layers_activation, power_on_z_score):
        y = df_aged['0'].values
        x = np.array([df_aggregate['phase_1'].values+y,
                    df_aggregate['phase_2'].values+y,
                    df_aggregate['phase_3'].values+y]).transpose()

        x = np.expand_dims(x,0)
        y = np.expand_dims(np.expand_dims(y,0),-1)

        ratio = 0.75

        #last 25% as validation dataset
        x_val = x[:,int(x.shape[1]*ratio):,:]
        y_val = y[:,int(y.shape[1]*ratio):,:]

        #first 75% as validation dataset
        x_train = x[:,:int(x.shape[1]*ratio),:]
        y_train = y[:,:int(y.shape[1]*ratio),:]
        # print('x_train shape:', x_train.shape, '\ty_train shape:', y_train.shape, '\tx_val shape:', x_val.shape, '\ty_val shape:',y_val.shape)

        #standerdizing
        target_std = x_train.std()
        target_mean = x_train.mean()
        x_val = (x_val-x.mean())/x_train.std()
        y_val = (y_val)/y_train.std()

        y_train_std = y_train.std()

        print('x_train mean:', x_train.mean(), '\nx_train std:', x_train.std(), '\ny_train std:', y_train.std())

        y_train = (y_train)/y_train.std()
        x_train = (x_train-x_train.mean())/x_train.std()

        appliance_model = make_main_model(x_train, y_train, depth = depth, n_layers = n_layers, kernel_size = kernel_size, dilation_size = dilation_size, middle_layers_activation = middle_layers_activation)
        return (x_val, y_val, appliance_model, target_std, target_mean, y_train_std)


    x_val, y_val, appliance_model, target_std, target_mean, y_train_std = load_model_for_appliance(
        depth = 16,
        n_layers = 6,
        kernel_size = 3,
        dilation_size = None,
        middle_layers_activation = 'relu',
        power_on_z_score = 0.25
    )


    three_ph_fridge_1_best_weights = os.path.join(current_directory, "weights", "3_phase_fridge_1_initial", "3_phase_fridge_1_weights.40-0.03.hdf5")


    appliance_model.load_weights(three_ph_fridge_1_best_weights)


    appliance_model.summary()


    def aging_adaptaion(inputs_1,input_2):
        x = tf.keras.layers.Concatenate()([inputs_1, input_2[0], input_2[1]])
        x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
        x = tf.keras.layers.Dense(64)(x)
        x = tf.keras.layers.Dense(64)(x)
        x = tf.keras.layers.Dense(32)(x)
        x = tf.keras.layers.Dense(1)(x)
        x =  wavenet(
            n_layers = n_layers,
            kernel_size = kernel_size,
            dialtion_base = kernel_size,
            out_channels = 32,
            causal = False,
            base_activation = 'relu',
            middle_layers_activation = 'relu',
            name = 'regression_model_aging')(x)
        return x


    inputs_1 = tf.keras.Input(shape=(None,3))
    inputs_2 = appliance_model(inputs_1)
    outputs = aging_adaptaion(inputs_1, inputs_2)
    aging_model = tf.keras.Model(inputs=inputs_1,outputs=outputs, name='aging_model')
    aging_model.compile(optimizer= tf.keras.optimizers.Nadam(initial_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss='mse', metrics=['mae'])

    if device_age == 7:
        # <keras.src.engine.input_layer.InputLayer at 0x7ddad0803ca0>,
        aging_model.layers[0].trainable = False
        # <keras.src.engine.functional.Functional at 0x7ddab730ab60>,
        aging_model.layers[1].trainable = False
        # <keras.src.layers.merging.concatenate.Concatenate at 0x7ddab713c9d0>,
        aging_model.layers[2].trainable = False


        # <keras.src.layers.rnn.lstm.LSTM at 0x7ddab71cb220>,
        aging_model.layers[3].trainable = False
        # <keras.src.layers.core.dense.Dense at 0x7ddab7163af0>,
        aging_model.layers[4].trainable = False
        # <keras.src.layers.core.dense.Dense at 0x7ddad9fa7b20>,
        aging_model.layers[5].trainable = True
        # <keras.src.layers.core.dense.Dense at 0x7ddad08008b0>,
        aging_model.layers[6].trainable = True
        # <keras.src.layers.core.dense.Dense at 0x7ddace552b90>,
        aging_model.layers[7].trainable = True

        # <__main__.wavenet at 0x7ddace5502b0>]
        aging_model.layers[8].trainable = True
    
    else:
        # <keras.src.engine.input_layer.InputLayer at 0x7ddad0803ca0>,
        aging_model.layers[0].trainable = False
        # <keras.src.engine.functional.Functional at 0x7ddab730ab60>,
        aging_model.layers[1].trainable = False
        # <keras.src.layers.merging.concatenate.Concatenate at 0x7ddab713c9d0>,
        aging_model.layers[2].trainable = False


        # <keras.src.layers.rnn.lstm.LSTM at 0x7ddab71cb220>,
        aging_model.layers[3].trainable = False
        # <keras.src.layers.core.dense.Dense at 0x7ddab7163af0>,
        aging_model.layers[4].trainable = False
        # <keras.src.layers.core.dense.Dense at 0x7ddad9fa7b20>,
        aging_model.layers[5].trainable = False
        # <keras.src.layers.core.dense.Dense at 0x7ddad08008b0>,
        aging_model.layers[6].trainable = True
        # <keras.src.layers.core.dense.Dense at 0x7ddace552b90>,
        aging_model.layers[7].trainable = True

        # <__main__.wavenet at 0x7ddace5502b0>]
        aging_model.layers[8].trainable = False

    aging_model.summary()

    if device_age == 0 or device_age == 1 or device_age == 2 or device_age == 3:
        weight_path = os.path.join(current_directory, "weights", "3_phase_fridge_1_year_3", "0_1_2_3_working_base_model_0_1_2_weight.h5")
        aging_model.load_weights(weight_path)
    else:
        print("im in")
        weight_path = os.path.join(current_directory, "weights", "3_phase_fridge_1_rest", f"weight_{device_age}.h5")
        print("\nloaded weight:",weight_path)  
        aging_model.load_weights(weight_path)  
        print("im out")



    def data_pipeline(year):

        y = df_aged[year].values
        x = np.array([df_aggregate['phase_1'].values+y,
                    df_aggregate['phase_2'].values+y,
                    df_aggregate['phase_3'].values+y]).transpose()

        x = np.expand_dims(x,0)
        y = np.expand_dims(np.expand_dims(y,0),-1)

        ratio = 0.75

        #last 25% as validation dataset
        x_val = x[:,int(x.shape[1]*ratio):,:]
        y_val = y[:,int(y.shape[1]*ratio):,:]

        #first 75% as validation dataset
        x_train = x[:,:int(x.shape[1]*ratio),:]
        y_train = y[:,:int(y.shape[1]*ratio),:]
        print('x_train shape:', x_train.shape, '\ty_train shape:', y_train.shape, '\tx_val shape:', x_val.shape, '\ty_val shape:',y_val.shape)

        #standerdizing
        target_std = x_train.std()
        target_mean = x_train.mean()
        x_val = (x_val-x.mean())/x_train.std()
        y_val = (y_val)/y_train.std()

        y_train_std = y_train.std()

        print('x_train mean:', x_train.mean(), '\nx_train std:', x_train.std(), '\ny_train std:', y_train.std())

        y_train = (y_train)/y_train.std()
        x_train = (x_train-x_train.mean())/x_train.std()

        return x_train, y_train, x_val, y_val, target_std, target_mean, y_train_std


    x_train_dict = {}
    y_train_dict = {}
    x_val_dict = {}
    y_val_dict = {}
    x_train_std_dict = {}
    x_train_mean_dict = {}
    y_train_std_dict = {}

    for i in df_aged.columns:
        print('\nyear:',i)
        x_train, y_train, x_val, y_val, x_train_std, x_train_mean, y_train_std = data_pipeline(i)
        x_train_dict[i] = x_train
        y_train_dict[i] = y_train
        x_val_dict[i] = x_val
        y_val_dict[i] = y_val
        x_train_std_dict[i] = x_train_std
        x_train_mean_dict[i] = x_train_mean
        y_train_std_dict[i] = y_train_std

    x_train =[]
    y_train =[]
    x_val =[]
    y_val =[]
    upper_bound = x_train_dict[list(x_train_dict.keys())[0]].shape[1]
    counter = sample_size
    while counter<upper_bound:
        for i in x_train_dict.keys():
            x_train.append(x_train_dict[i][:,counter-sample_size:counter,:])
            y_train.append(y_train_dict[i][:,counter-sample_size:counter,:])
        counter+=sample_size

    upper_bound = x_train_dict[list(x_val_dict.keys())[0]].shape[1]
    counter = sample_size
    while counter<upper_bound:
        for i in x_val_dict.keys():
            x_val.append(x_val_dict[i][:,counter-sample_size:counter,:])
            y_val.append(y_val_dict[i][:,counter-sample_size:counter,:])
        counter+=sample_size



    x_train = np.concatenate(x_train, axis=1)
    y_train = np.concatenate(y_train, axis=1)
    x_val = np.concatenate(x_val, axis=1)
    y_val = np.concatenate(y_val, axis=1)


    def gen(x,y,window_size = 4200):
        def blank():
            mean = y.mean()
            i = window_size
            while i < x.shape[1]:
                yield x[:,i-window_size:i,:] , y[:,i-window_size:i,:]
                i += window_size*skiping_factor
        return blank

    train_gen = gen(x_train,y_train)
    val_gen = gen(x_val, y_val)

    train_gen = tf.data.Dataset.from_generator(
        train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,1), dtype=tf.float32))).repeat()

    val_gen = tf.data.Dataset.from_generator(
        val_gen,
        output_signature=(
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,1), dtype=tf.float32))).repeat()

    agg_std = 2244.651761243073
    agg_mean = 2170.743671588378

    output ={}
    def aged_predictions(window):
        years = len(df_aged.columns)
        for year in range(0,years) :
            x_val, y_val = x_val_dict[str(year)], y_val_dict[str(year)]
            pred = aging_model(x_val[:,window[0]:window[1],:]).numpy()

            target = y_val_dict['0']
            n1 = window[0]
            n2 = window[1]

            print(f'Aggregate Power for Year {year}')
            print("Aggregate consumption (Phase 1):",(x_val[0, n1:n2, 0]*agg_std)+agg_mean)
            print('Aggregate consumption (Phase 2):',(x_val[0, n1:n2, 1]*agg_std)+agg_mean)
            print('Aggregate consumption (Phase 3)',(x_val[0, n1:n2, 2]*agg_std)+agg_mean)

            target_std = y_train_std_dict[f'{year}']
            target_std_year0 =  y_train_std_dict['0']
            error = y_val[0,n1:n2,0]*target_std-pred[0,:,0]*target_std

            print(f'Disggregate Power for Year {year}')
            print(f'Target(year 0)',target[0,n1:n2,0]*target_std_year0)
            print(f'Target(year {year})',y_val[0,n1:n2,0]*target_std)
            print(f"Prediction(year {year})",pred[0,:,0]*target_std)

            if year == device_age:
                output = {
                'aggregated_power_phase_1': list((x_val[0, n1:n2, 0]*agg_std)+agg_mean),
                'aggregated_power_phase_2': list((x_val[0, n1:n2, 1]*agg_std)+agg_mean),
                'aggregated_power_phase_3': list((x_val[0, n1:n2, 2]*agg_std)+agg_mean),
                'disaggregated_power_target':list(y_val[0,n1:n2,0]*target_std),
                'disaggregated_power_prediction': list(pred[0,:,0]*target_std)
                }



        return output

    output = aged_predictions(window =[start_point, end_point])
    print(output)
    return output