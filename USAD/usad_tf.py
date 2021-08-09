import tensorflow as tf


class Encoder(tf.keras.Model) :
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape = (in_size)),
            tf.keras.layers.Dense(int(in_size/2), activation = "relu"),
            tf.keras.layers.Dense(int(in_size/4), activation = "relu"),
            tf.keras.layers.Dense(latent_size, activation = "relu")
        ])
    
    def call(self, x) :
        z = self.model(x)
        return z
    
class Decoder(tf.keras.Model) :
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape = (latent_size)),
            tf.keras.layers.Dense(int(out_size/4), activation = "relu"),
            tf.keras.layers.Dense(int(out_size/2), activation = "relu"),
            tf.keras.layers.Dense(out_size, activation = "sigmoid")
        ])
    
    def call(self, x) :
        w = self.model(x)
        return w
    
class UsadModel(tf.keras.Model):
    def __init__(self, w_size, z_size, alpha = .5, beta = .5):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)
        
        self.latent_vector = self.encoder(self.encoder.model.inputs)
        self.ae_output1 = self.decoder1(self.latent_vector)
        self.ae_output2 = self.decoder2(self.latent_vector)

        self.ae_model1 = tf.keras.Model(inputs = self.encoder.model.inputs, outputs = self.ae_output1)
        self.ae_model2 = tf.keras.Model(inputs = self.encoder.model.inputs, outputs = self.ae_output2)
        
        self.optimizer = tf.keras.optimizers.Adam()
        
        self.alpha, self.beta = alpha, beta
        
    def evaluate(self, val_loader, n):
        outputs = [self.validation_step(batch, n) for batch in val_loader]
        return self.validation_epoch_end(outputs)
    
    def testing(self, test_loader):
        results=[]
        for batch, _ in test_loader:
            w1=self.ae_model1(batch)
            w2=self.ae_model2(w1)
            results.append(self.alpha*np.mean((batch-w1).numpy()**2, axis = 1)+self.beta*np.mean((batch-w2).numpy()**2, axis = 1))
        return results
        
    def call(self, x):
        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return w1, w2, w3
    
    def loss_fn(self, batch, n) :
        loss1 = 1/n*tf.reduce_mean(tf.square(batch-self.w1)) + (1-1/n)*tf.reduce_mean(tf.square(batch-self.w3))
        loss2 = 1/n*tf.reduce_mean(tf.square(batch-self.w2)) + (1-1/n)*tf.reduce_mean(tf.square(batch-self.w3))
        
        return loss1, loss2
    
    def training(self, train_loader, val_loader, num_epochs):
        for n in range(num_epochs): 
            n += 1
            
            loss1_list, loss2_list = [], []
            self.history = []
            
            # Iterate over the batches of a dataset.
            for x_batch_train, y_batch_train in train_loader:
                with tf.GradientTape() as ae1_tape, tf.GradientTape() as ae2_tape:
                    self.z = self.encoder(x_batch_train)
                    self.w1 = self.decoder1(self.z)
                    self.w2 = self.decoder2(self.z)
                    self.w3 = self.decoder2(self.encoder(self.w1))

                    # Loss value for this minibatch
                    loss1, loss2 = self.loss_fn(x_batch_train, n)
                    
                    # Add extra losses created during this forward pass:
                    #loss_value += sum(model.losses)

                grads_ae1 = ae1_tape.gradient(loss1, self.ae_model1.trainable_weights)
                self.optimizer.apply_gradients(zip(grads_ae1, self.ae_model1.trainable_weights))
                grads_ae2 = ae2_tape.gradient(loss2, self.ae_model2.trainable_weights)
                self.optimizer.apply_gradients(zip(grads_ae2, self.ae_model2.trainable_weights))
                loss1_list.append(loss1)
                loss2_list.append(loss2)
                
            #print("Epoch [{}], train_loss1: {:.4f}, train_loss2: {:.4f}".format(n, np.mean(loss1_list), np.mean(loss2_list)))
            result = self.evaluate(val_loader, n)
            self.epoch_end(n, result)
            self.history.append(result)
            
    def validation_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1/n*tf.reduce_mean(tf.square(batch-w1)) + (1-1/n)*tf.reduce_mean(tf.square(batch-w3))
        loss2 = 1/n*tf.reduce_mean(tf.square(batch-w2)) + (1-1/n)*tf.reduce_mean(tf.square(batch-w3))
        
        return {'val_loss1': loss1, 'val_loss2': loss2}
    
    def validation_epoch_end(self, outputs):
        batch_losses1 = [x['val_loss1'] for x in outputs]
        epoch_loss1 = tf.reduce_mean(batch_losses1)
        batch_losses2 = [x['val_loss2'] for x in outputs]
        epoch_loss2 = tf.reduce_mean(batch_losses2)
        return {'val_loss1': epoch_loss1.numpy(), 'val_loss2': epoch_loss2.numpy()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))
        
        
class UsadModel_AGM(tf.keras.Model):
    def __init__(self, w_size, z_size, alpha = .5, beta = .5):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)
        
        self.latent_vector = self.encoder(self.encoder.model.inputs)
        self.ae_output1 = self.decoder1(self.latent_vector)
        self.ae_output2 = self.decoder2(self.latent_vector)

        self.ae_model1 = tf.keras.Model(inputs = self.encoder.model.inputs, outputs = self.ae_output1)
        self.ae_model2 = tf.keras.Model(inputs = self.encoder.model.inputs, outputs = self.ae_output2)
        
        self.optimizer = tf.keras.optimizers.Adam()
        
        self.alpha, self.beta = alpha, beta
        
    def evaluate(self, val_loader, n):
        outputs = [self.validation_step(batch, n) for batch in val_loader]
        return self.validation_epoch_end(outputs)

    def testing(self, test_loader):
        results=[]
        for batch, _ in test_loader:
            w1=self.ae_model1(batch)
            w2=self.ae_model2(w1)
            results.append(self.alpha*np.mean((batch-w1).numpy()**2, axis = 1)+self.beta*np.mean((batch-w2).numpy()**2, axis = 1))
        return results
        
    def call(self, x):
        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return w1, w2, w3
    
    def loss_fn(self, batch, n) :
        loss1 = 1/n*tf.reduce_mean(tf.square(batch-self.w1)) + (1-1/n)*tf.reduce_mean(tf.square(batch-self.w3))
        loss2 = 1/n*tf.reduce_mean(tf.square(batch-self.w2)) + (1-1/n)*tf.reduce_mean(tf.square(batch-self.w3))
        
        return loss1, loss2
    
    def training(self, train_loader, val_loader, num_epochs):
        self.history = []
        
        for n in range(num_epochs): 
            n += 1
            
            loss1_list, loss2_list, loss3_list = [], [], []
            
            # Iterate over the batches of a dataset.
            for x_batch_train, y_batch_train in train_loader:
                with tf.GradientTape() as ae1_tape, tf.GradientTape() as ae2_tape, tf.GradientTape() as pg_tape:
                    self.z = self.encoder(x_batch_train)
                    self.w1 = self.decoder1(self.z)
                    self.w2 = self.decoder2(self.z)
                    self.w3 = self.decoder2(self.encoder(self.w1))
                    
                    real_recon1 = x_batch_train-self.w1
                    real_recon2 = x_batch_train-self.w2
                    fake_recon = x_batch_train-self.w3

                    # Loss value for this minibatch
                    loss1, loss2 = self.loss_fn(x_batch_train, n)
                    
                    pg_advantage = tf.stop_gradient(tf.reduce_mean(tf.square(fake_recon))-tf.reduce_mean(tf.square(real_recon2)))
                    loss3 = -tf.reduce_mean(tf.math.log(self.w3+1e-6) * pg_advantage)
        
                    
                    # Add extra losses created during this forward pass:
                    #loss_value += sum(model.losses)

                grads_ae1 = ae1_tape.gradient(loss1, self.ae_model1.trainable_weights)
                self.optimizer.apply_gradients(zip(grads_ae1, self.ae_model1.trainable_weights))
                grads_ae2 = ae2_tape.gradient(loss2, self.ae_model2.trainable_weights)
                self.optimizer.apply_gradients(zip(grads_ae2, self.ae_model2.trainable_weights))
                grads_ae3 = pg_tape.gradient(loss3, self.ae_model2.trainable_weights)
                self.optimizer.apply_gradients(zip(grads_ae3, self.ae_model2.trainable_weights))
                
                loss1_list.append(loss1)
                loss2_list.append(loss2)
                loss3_list.append(loss3)
                
            result = self.evaluate(val_loader, n)
            self.epoch_end(n, result)
            self.history.append(result)
                
            #print("Epoch [{}], train_loss1: {:.4f}, train_loss2: {:.4f}, train_loss3: {:.4f}".format(n, np.mean(loss1_list), np.mean(loss2_list), np.mean(loss3_list)))
            
    def validation_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        
        real_recon1 = batch-w1
        real_recon2 = batch-w2
        fake_recon = batch-w3
                
        loss1 = 1/n*tf.reduce_mean(tf.square(batch-w1)) + (1-1/n)*tf.reduce_mean(tf.square(batch-w3))
        loss2 = 1/n*tf.reduce_mean(tf.square(batch-w2)) + (1-1/n)*tf.reduce_mean(tf.square(batch-w3))
        
        pg_advantage = tf.reduce_mean(tf.square(fake_recon))-tf.reduce_mean(tf.square(real_recon2))
        loss3 = -tf.reduce_mean(tf.math.log(w3+1e-6) * pg_advantage)
        
        return {'val_loss1': loss1, 'val_loss2': loss2, 'val_loss3': loss3}
    
    def validation_epoch_end(self, outputs):
        batch_losses1 = [x['val_loss1'] for x in outputs]
        epoch_loss1 = tf.reduce_mean(batch_losses1)
        batch_losses2 = [x['val_loss2'] for x in outputs]
        epoch_loss2 = tf.reduce_mean(batch_losses2)
        batch_losses3 = [x['val_loss3'] for x in outputs]
        epoch_loss3 = tf.reduce_mean(batch_losses3)
        return {'val_loss1': epoch_loss1.numpy(), 'val_loss2': epoch_loss2.numpy(), 'val_loss3': epoch_loss3.numpy()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2'], result['val_loss3']))