import sys

from FeatureCloud.app.engine.app import AppState, app_state, Role
from typing import TypedDict, Union, List, Type
from torch.utils.data import Dataset, DataLoader
import utils
import torch
import bios
import importlib.util
import io




from fc_quantization.Compress.compress import QuantAppState
import fc_quantization.Compress.utils as pf

INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'



@app_state('initial')
class InitialState(QuantAppState):

    def register(self):
        # Register transition for local update
        self.register_transition('local_update',label="Broadcast initial weights")


    def run(self) :

        # Reading configuration file
        self.log('Reading configuration file ...')

        # Loading configuration from file
        config = bios.read(f'{INPUT_DIR}/config.yml')

        max_iterations = config['max_iter']
        self.store('iteration', 0)
        self.store('max_iterations', max_iterations)
        self.store('learning_rate', config['learning_rate'])
        self.store('epochs', config['epochs'])
        self.store('batch_size', config['batch_size'])
        self.store('backend', config['backend'])
        self.store('quant_type', config['quant_type'])

        train_dataset_path = f"{INPUT_DIR}/{config['train_dataset']}"
        test_dataset_path = f"{INPUT_DIR}/{config['test_dataset']}"
        train_dataset = torch.load(train_dataset_path)
        test_dataset = torch.load(test_dataset_path)
        self.store('train_dataset', train_dataset)
        self.store('test_dataset', test_dataset)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.load('batch_size'), shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.load('batch_size'), shuffle=False)
        self.store('train_loader', train_loader)
        self.store('test_loader', test_loader)


        self.log('Done reading configuration.')

        # Loading and preparing initial model
        self.log('Preparing initial model ...')
        model_path = f"{INPUT_DIR}/{config['model']}"

        # Loading model from file
        spec = importlib.util.spec_from_file_location("model_module", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        model_class_name = config.get('model_class', 'Model')
        model = getattr(model_module, model_class_name)()
        # Storing model
        model.quant = torch.quantization.QuantStub()
        model.dequant = torch.quantization.DeQuantStub()
        self.store('model', model)

        reference_model = getattr(model_module, model_class_name)()
        reference_model.quant = torch.quantization.QuantStub()
        reference_model.dequant = torch.quantization.DeQuantStub()
        self.store('reference_model', reference_model)



        self.log('Transition to local state ...')

        if self.is_coordinator:
            # Broadcasting initial weights to participants
            self.broadcast_data([pf.get_weights(model),False], send_to_self=False)
            self.store('received_data', pf.get_weights(model))

        return 'local_update'





@app_state('local_update', Role.BOTH)
class LocalUpdate(QuantAppState):
    def register(self):
        # Registering transitions for local update
        self.register_transition('aggregation', Role.COORDINATOR, label="Gather local models")
        self.register_transition('local_update', Role.PARTICIPANT, label="Wait for global model")
        self.register_transition('terminal',label="Terminate process")


    def run(self):
        # Running local update process

        iteration = self.load('iteration')
        self.log(f'ITERATION  {iteration}')
        model = self.load('model')
        stop_flag = False
        if self.is_coordinator:
            received_data = self.load('received_data')
        else:
            received_data, stop_flag = self.await_data(unwrap=True)


        if stop_flag:
            self.log('Stopping')
            return 'terminal'

        # Receive global model from coordinator
        self.log('Receive model from coordinator')

        pf.set_weights(model,received_data)

        # Receive dataframe
        train_loader = self.load('train_loader')
        epochs = self.load('epochs')
        learning_rate = self.load('learning_rate')
        test_loader = self.load('test_loader')
        backend = self.load('backend')
        quant_type = self.load('quant_type')

        self.configure_quant(model=model, backend=backend, quant_type=quant_type )

        # Train local model
        if quant_type == 'post_static':
            self.log('Training local model ...')
            utils.train(model, train_loader, epochs=epochs, learning_rate=learning_rate)
            prep_model = pf.prep_post_static_quant(model, train_loader, backend)
        else:
            prep_model = pf.prepare_qat(model,backend)
            utils.train(prep_model, train_loader, epochs, learning_rate) # use custom training function

        #reconfigure quant
        self.configure_quant(model=prep_model, backend=backend, quant_type=quant_type )

        self.send_data_to_coordinator(prep_model, use_quant=True, use_smpc=False, use_dp=False)

        # Test quantized model
        utils.test(model, test_loader)

        iteration += 1
        self.store('iteration', iteration)

        if stop_flag:
            return 'terminal'

        if self.is_coordinator:
            return 'aggregation'

        else:
            return 'local_update'

@app_state('aggregation', Role.COORDINATOR)
class AggregateState(QuantAppState):

    def register(self):
        # Registering transitions for aggregation state
        self.register_transition('local_update', Role.COORDINATOR, label="Broadcast global model")
        self.register_transition('terminal', Role.COORDINATOR, label="Terminate process")

    def run(self) :
        # Running aggregation process
        self.log(f'Aggregating Data ...')
        # Gathering and averaging data
        data = self.gather_data(use_quant=True, is_json=False, use_smpc=False, use_dp=False, memo=None)
        self.log(f'Averaging Data ...')

        global_averaged_weights = utils.average_weights_quant(data)

        stop_flag = False
        if self.load('iteration') >= self.load('max_iterations'):
            stop_flag = True

        # Set averaged_weights as new global model
        self.store('received_data', global_averaged_weights)
        new_model= self.load('model')
        pf.set_weights(new_model, global_averaged_weights)

        # Broadcasting global model
        self.log('Broadcasting global model ...')
        self.broadcast_data([global_averaged_weights, stop_flag], send_to_self=False)


        if stop_flag:
            return 'terminal'

        return 'local_update'

