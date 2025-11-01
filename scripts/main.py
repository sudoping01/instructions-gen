# main.py
import asyncio
import os

from processor import InstructionDatasetPreprocessor, setup_logger
import settings

async def process_single_dataset(input_file: str, output_file: str):
    translator = InstructionDatasetPreprocessor(
        api_key=settings.API_KEY,
        max_workers=settings.MAX_WORKERS,          
        concurrent_batches=settings.CONCURRENT_BATCHES,    
        enable_caching=True
    )
    try:
        results = await translator.translate_dataset_concurrent(
            input_file, 
            output_file, 
            batch_size=settings.BATCH_SIZE,           
            max_concurrent_batches=settings.CONCURRENT_BATCHES
        )
        return results
    finally:
        await translator.close()


async def process_multiple_datasets():
    logger = setup_logger()
    files = os.listdir(settings.DATASETS_FOLDER)
    logger.info(f"Datasets found in {settings.DATASETS_FOLDER}: {files}")

    translation_tasks = []

    for file in files:
        if file.endswith(".jsonl"):
            output_file = f"bam_{file}"
            translation_tasks.append((file, output_file))
    
    logger.info(f" Starting {len(translation_tasks)} dataset translations sequentially")
    
    for i, (input_file, output_file) in enumerate(translation_tasks, 1):
        logger.info(f" Task {i}: Starting {input_file} -> {output_file}")
        
        try:
            result = await process_single_dataset(input_file, output_file)
            logger.info(f"Task {i}: Completed {input_file} ({len(result) if result else 0} conversations)")
        except Exception as e:
            logger.error(f" Task {i}: Failed {input_file}: {e}")
        
        if i < len(translation_tasks):
            await asyncio.sleep(5)


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Starting Concurrent Instruction Dataset Processing")
    
    try:
        asyncio.run(process_multiple_datasets())
        
    except KeyboardInterrupt:
        logger.warning("PROCESSING INTERUPTED")
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        raise
    finally:
        logger.info("SYSTEM SHUTDOWN")