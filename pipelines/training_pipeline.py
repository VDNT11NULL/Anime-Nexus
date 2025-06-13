from config.paths_config import *
# from src.data_ingestion import DataIngestion
from utils.common_functions import *
from src.data_processing import *
from src.model_training import ModelTrainer

if __name__=='__main__':

    ### INstead use DVC for data versioning    
    # data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    # data_ingestion.run()

    input_file = ANIMELIST_CSV  # rating data file
    output_dir = PROCESSED_DIR  # directory to save processed data
    
    processor = DataProcessor(input_file, output_dir)
    processor.run()


    trainer = ModelTrainer(data_path=PROCESSED_DIR)
    try:
        metrics = trainer.run()
        print(f"Training metrics: {metrics}")       
    except CustomException as e:
        logger.error(f"CustomException occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")