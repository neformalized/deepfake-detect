from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

def create(input_shape, output_shape):
    
    model = Sequential()
    
    ###
    
    model.add(Input(shape=input_shape))
    
    model.add(Conv3D(8, (4, 4, 4), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    model.add(Conv3D(16, (4, 4, 4), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    model.add(Conv3D(16, (4, 4, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    
    model.add(Dense(output_shape, activation='sigmoid'))
    
    ###
    
    optimizer = Adam(learning_rate = 0.0005)
    model.compile(optimizer = optimizer, loss = "binary_crossentropy")
    
    ###
    
    model.summary()
    
    ###
    
    return model
#