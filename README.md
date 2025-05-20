# Translation_model_hindi

Deep learning_assignment3
please install the dependencies before running the program other wise it may give error .I have added the requirement.txt file in the same folder. Please download the code from github and extract it .Then install the dependencies and run trainRNN.py .Note that all the other necessary module should be there in the same folder as with trainRNN.py as it import all the other classes from different file . If you want to run it in kaggle/colab it will also run normally as command promt.



pip install requirement.txt


python trainRNN.py -p path

My code is very much flexible to add in command line arguments . 
I am adding the list of possible argument below for your reference.
Please try to run this on local PC or from command promt by ensuring all the libraries in requirements.txt are already installed in your system.

Because in google colab this might give problem . python trainRNN.py -p path to the folder where data reside.

For example C:\Users\USER\Downloads\aksharantar_sampled\aksharantar_sampled\tel this where my train data,validation data,test data reside then you just given till C:\Users\Elitebook\nashra Dropbox\Nashra\PC\Downloads\dakshina_dataset_v1.0\dakshina_dataset_v1.0\hi , 

program will add train and val portion itself. 
My model is prepare to work on Hindi language so giving location to other dataset for example asami or urdu give error .

This script supports a range of command-line arguments for configuring and training a deep learning model. 


The --project_name or -wp argument, which defaults to Assignment3_DL, specifies the project name for logging runs to Weights & Biases (WandB). 


The --wandb_entity or -we flag identifies the WandB entity, such as amaannashra012-iit-madras-alumni-assosiation, for organizing experiment tracking.
The training duration is set using --epochs or -e, with a default of 15, while --batch_size or -b determines the batch size used during training, defaulting to 64.

The model's architecture can be customized via --cell_type or -es, allowing a choice among LSTM, RNN, or GRU (default is LSTM). The --embedding_size or -es and --hidden_size or -hs arguments set the size of the encoder and decoder embeddings and hidden states, respectively, both defaulting to 256.

The number of layers in the encoder and decoder are specified with --encoder_num_layer or -enl (default 2) and --decoder_num_layer or -dnl (default 1). Dropout is controlled using --dropout_rate or -do, with a default of 0.3 to prevent overfitting. 

The --bidirectional or -bd flag enables the use of bidirectional RNNs if set to True. The --attention or -a argument (1 for enabled, 0 for disabled) determines whether an attention mechanism is applied during decoding. 

A heatmap visualization of attention weights can be optionally generated using --plot_heatmap or -hp (1 to enable, 0 to disable). Finally, the --path or -p argument is mandatory and specifies the directory path where the training data is located.


This will run my best model which i get by validation accuracy. after that it will create a log in a project named CS6910-assignment12 by default until user dont specify project name.


# "parameters":
    {
        'hidden_size': {"values":[256]},
        'batch_size': {"values":[64]},
        'encoder_num_layers': {"values":[2]},
        'decoder_num_layers': {"values":[1]},
        'embedding_size': {"values":[128]},
        'drop_out': {"values":[0.3]},
        'epochs':{"values":[15]},
        'cell_type':{"values":["LSTM"]},
        'bidirectional':{"values":[True]}
}
      
    }```
Now if you want to change the number of layer I just have to execute the following the command.
python trainRNN.py -e 6

this will change the number of epoch to 6. Similarly we can use other commands as well.
My code plots to  wandb so please provide wandb key and project name and wandb entity appropriately so it can plot logs to wandb. 
