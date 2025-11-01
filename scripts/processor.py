
from typing import List, Dict, Optional, Tuple

from openai import AsyncOpenAI
from datetime import datetime
from pathlib import Path

import asyncio
import json
import os
import time
import hashlib
import pickle

import settings  # Import settings module
from logger import setup_logger



class InstructionDatasetPreprocessor:
    def __init__(self, 
                 api_key: Optional[str] = settings.API_KEY, 
                 model: str = settings.MODEL_NAME, 
                 max_workers: int = settings.MAX_WORKERS,  
                 enable_caching: bool = True,
                 concurrent_batches: int = settings.CONCURRENT_BATCHES,
                 timeout: float = settings.TIMEOUT,
                 max_retries: int = settings.MAX_RETRIES,
                 base_url: str = settings.BASE_URL, 
                 rate_limiter: int = settings.RATE_LIMITER
                 ):  

        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers
        self.concurrent_batches = concurrent_batches
        self.enable_caching = enable_caching
        self.logger = setup_logger()
        

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries
        )
        
        self.semaphore = asyncio.Semaphore(max_workers)
        self.batch_semaphore = asyncio.Semaphore(concurrent_batches)  
        self.rate_limiter = asyncio.Semaphore(rate_limiter)  
        

        self.cache_hits = 0
        self.cache_misses = 0
        self.total_api_calls = 0
        self.total_api_time = 0.0
        self.failed_batches = []
        
        self.linguistic_resources = self.load_linguistic_resources()
        self.logger.info(f"PROCESS INSTANTIATION: {model}, {max_workers} workers, {concurrent_batches} concurrent batches")

    def load_linguistic_resources(self) -> str:
        cache_file = settings.CACHE_DIR / "linguistic_resources.pkl"
        if self.enable_caching and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    context = pickle.load(f)
                    self.logger.info("LOADING LINGUISTIC CONTENT: Loaded from cache ")
                    return context
            except Exception as e:
                self.logger.warning(f"LOADING LINGUISTIC CONTENT: Cache load failed: {e}")
        
        context_parts = []
        files_loaded = 0
        
        try:
            with open(settings.GLOSSARY_FILE, "r", encoding="utf-8") as f:
                glossary = json.load(f)
                context_parts.append(f"{settings.LANGUAGE_NAME} GLOSSARY:")
                context_parts.append(json.dumps(glossary, ensure_ascii=False, indent=2))
                files_loaded += 1
                self.logger.info(f"GLOSSARY LOADING: Loaded Successfully {settings.GLOSSARY_FILE}")
        except Exception as e:
            self.logger.warning(f"GLOSSARY LOADING: Failed to load {settings.GLOSSARY_FILE}: {e}")
        
        try:
            with open(settings.GRAMMAR_RULE_FILE, "r", encoding="utf-8") as f:
                grammar = json.load(f)
                context_parts.append("\nBAMBARA GRAMMAR RULES:")
                context_parts.append(json.dumps(grammar, ensure_ascii=False, indent=2))
                files_loaded += 1
                self.logger.info(f"GRAMMAR LOADING: Loaded Successfully {settings.GRAMMAR_RULE_FILE}")
        except Exception as e:
            self.logger.warning(f"GRAMMAR LOADING: Failed to load {settings.GRAMMAR_RULE_FILE}: {e}")

        try:
            with open(settings.CONLLU_FILE, "r", encoding="utf-8") as f:
                conllu_data = f.read()
                context_parts.append("\nBAMBARA LINGUISTIC EXAMPLES:")
                context_parts.append(conllu_data)
                files_loaded += 1
                self.logger.info(f"BAMBARA LINGUISTIC EXAMPLES LOADING: Loaded Successfully {settings.CONLLU_FILE} ({len(conllu_data)} chars)")
        except Exception as e:
            self.logger.warning(f"BAMBARA LINGUISTIC EXAMPLES LOADING: Failed to load {settings.CONLLU_FILE}: {e}")

        context = "\n".join(context_parts)
        

        if self.enable_caching:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(context, f)
                self.logger.info(f"LINGUISTIC RESOURCES CACHING: Cached {settings.LANGUAGE_NAME} context in {cache_file}")
            except Exception as e:
                self.logger.warning(f"LINGUISTIC RESOURCES CACHING: Context caching failed: {e}")
        
        self.logger.info(f"{settings.LANGUAGE_NAME} LINGUISTIC RESOURCES: loaded {files_loaded} files, {len(context)} chars")
        return context


    def _get_batch_cache_key(self, batch_data: List[Dict]) -> str:
        batch_str = json.dumps(batch_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(batch_str.encode()).hexdigest()

    async def _load_batch_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        if not self.enable_caching:
            return None
            
        cache_file = settings.CACHE_DIR / f"batch_{cache_key}.pkl"
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
                self.cache_hits += 1
                return result
        except Exception as e:
            self.logger.warning(f"BATCH CACHING: Cache load failed for {cache_key}: {e}")
            return None

    async def _save_batch_to_cache(self, cache_key: str, result: List[Dict]):
        if not self.enable_caching:
            return
            
        cache_file = settings.CACHE_DIR / f"batch_{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            self.logger.warning(f"BATCH CACHING: Cache save failed for {cache_key}: {e}")

    async def _translate_with_retry(self, batch_data: List[Dict], max_retries: int = settings.MAX_RETRIES) -> List[Dict]:        
        for attempt in range(max_retries):
            try:
                async with self.rate_limiter:
                    batch_json = json.dumps(batch_data, ensure_ascii=False, separators=(',', ':'))

                    system_prompt = f"""You are an expert translator specializing in {settings.LANGUAGE_NAME} language instruction datasets.

                                    LINGUISTIC RESOURCES:
                                    {self.linguistic_resources}

                                    TRANSLATION PROTOCOL:
                                    - Maintain exact JSON structure and formatting
                                    - Preserve all roles (system, user, assistant) 
                                    - Apply proper {settings.LANGUAGE_NAME} grammar from provided resources and your general knowledge 
                                    - Keep original tone and formality levels
                                    - Preserve formatting, numbers, special characters
                                    - Convert person names to appropriate {settings.LANGUAGE_NAME} names when appropriate
                                    - Use French code-switching for untranslatable terms
                                    - Return ONLY valid JSON arrays matching input structure

                                    Return the translated JSON array with same structure as input."""


                    user_prompt = f"Translate these conversations to {settings.LANGUAGE_NAME}:\n{batch_json}"

                    api_start_time = time.time()
                    self.total_api_calls += 1
                    
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=settings.MAX_TOKENS, 
                        temperature=settings.TEMPERATURE,
                        top_p=settings.TOP_P,
                        timeout=settings.BATCH_TIMEOUT,
                        response_format={"type": "json_object"}
                    )
                    
                    api_time = time.time() - api_start_time
                    self.total_api_time += api_time
                    
                    return await self._parse_response(response.choices[0].message.content.strip(), len(batch_data))
                        
            except Exception as e:
                wait_time = settings.RETRY_DELAY * (2 ** attempt)
                self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise


    async def _parse_response(self, response_content: str, expected_count: int) -> List[Dict]:
        try:
            if response_content.startswith('```'):
                start_idx = response_content.find('[')
                end_idx = response_content.rfind(']') + 1
                json_str = response_content[start_idx:end_idx]
            else:
                json_str = response_content
            
            translated_batch = json.loads(json_str)
            
            if not isinstance(translated_batch, list) or len(translated_batch) != expected_count:
                raise ValueError(f"Invalid response structure: expected {expected_count} items, got {len(translated_batch)}")
            
            results = []
            for item in translated_batch:
                if not isinstance(item, dict) or "messages" not in item:
                    raise ValueError("Invalid item structure in response")
                results.append({"messages": item["messages"]})
            
            return results
            
        except Exception as e:
            self.logger.error(f"Response parsing failed: {e}")
            return []


    async def translate_batch(self, conversations_batch: List[Tuple[int, Dict]]) -> List[Dict]:
        batch_size = len(conversations_batch)
        batch_ids = [str(idx) for idx, _ in conversations_batch]
        
        async with self.batch_semaphore:  
            async with self.semaphore:
                try:

                    batch_data = []
                    for idx, conversation in conversations_batch:
                        batch_data.append({
                            "id": idx,
                            "messages": conversation.get("messages", [])
                        })
                    

                    cache_key = self._get_batch_cache_key(batch_data)
                    cached_result = await self._load_batch_from_cache(cache_key)
                    
                    if cached_result:
                        self.logger.info(f"Cache HIT for batch {batch_ids[0]}-{batch_ids[-1]}")
                        return cached_result
                    
                    self.cache_misses += 1
                    self.logger.info(f"Processing batch {batch_ids[0]}-{batch_ids[-1]} ({batch_size} conversations)")

                    start_time = time.time()
                    results = await asyncio.wait_for(
                        self._translate_with_retry(batch_data),
                        timeout=settings.BATCH_TIMEOUT
                    )

                    batch_time = time.time() - start_time

                    if results:
                        await self._save_batch_to_cache(cache_key, results)
                    
                    avg_time = batch_time / len(results) if results else 0
                    self.logger.info(f"Batch {batch_ids[0]}-{batch_ids[-1]} completed: {len(results)}/{batch_size} in {batch_time:.2f}s ({avg_time:.2f}s/conv)")
                    
                    return results
                    
                except asyncio.TimeoutError:
                    self.logger.error(f"Batch {batch_ids[0]}-{batch_ids[-1]} timed out")
                    self.failed_batches.append((batch_ids, "timeout"))
                    return []
                except Exception as e:
                    self.logger.error(f"Batch {batch_ids[0]}-{batch_ids[-1]} failed: {e}")
                    self.failed_batches.append((batch_ids, str(e)))
                    return []


    def read_jsonl(self, file_path: str) -> List[Dict]:
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing JSONL line: {e}")
                        continue
        except FileNotFoundError:
            self.logger.error(f"File {file_path} not found!")
            return []
        return data

    def save_jsonl(self, data: List[Dict], output_file: str):
        try:
            temp_file = f"{output_file}.tmp"
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n')
            
            os.rename(temp_file, output_file)
            self.logger.info(f"Saved {len(data)} conversations to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving to {output_file}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)


    def _save_checkpoint(self, checkpoint_file: str, results: List[Dict]):
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}")


    def _load_checkpoint(self, checkpoint_file: str) -> List[Dict]:
        try:
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Checkpoint load failed: {e}")
            return []

    async def translate_dataset_concurrent(self, 
                                         input_file: str, 
                                         output_file: str, 
                                         batch_size: int = settings.BATCH_SIZE,
                                         max_concurrent_batches: int = settings.CONCURRENT_BATCHES) -> List[Dict]:

        self.logger.info(f"CONCURRENT translation: {input_file} -> {output_file}")
        start_time = time.time()
        
        dataset = self.read_jsonl(input_file)
        if not dataset:
            return []
        
        total_conversations = len(dataset)
        self.logger.info(f"Processing {total_conversations} conversations (batch_size={batch_size}, concurrent_batches={max_concurrent_batches})")
        
        checkpoint_file = f"{output_file}.checkpoint"
        all_results = []
        completed_indices = set()
        
        if os.path.exists(checkpoint_file):
            all_results = self._load_checkpoint(checkpoint_file)
            completed_indices = set(range(len(all_results)))
            self.logger.info(f"Resumed from checkpoint: {len(all_results)} completed")
        
        conversations_with_idx = [(idx + 1, conv) for idx, conv in enumerate(dataset) if idx not in completed_indices]
        batches = [conversations_with_idx[i:i + batch_size] for i in range(0, len(conversations_with_idx), batch_size)]
        
        total_batches = len(batches)
        self.logger.info(f"Processing {total_batches} batches")
        
        batch_groups = [batches[i:i + max_concurrent_batches] for i in range(0, len(batches), max_concurrent_batches)]
        
        for group_idx, batch_group in enumerate(batch_groups, 1):
            group_start_time = time.time()
            
            batch_tasks = [self.translate_batch(batch) for batch in batch_group]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, list):
                    all_results.extend(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Batch failed: {result}")
            
            group_time = time.time() - group_start_time
            self.logger.info(f"Group {group_idx}/{len(batch_groups)} completed in {group_time:.1f}s")
            total_completed = len(all_results)
            progress = (total_completed / total_conversations) * 100
            

            elapsed_time = time.time() - start_time
            if total_completed > 0:
                estimated_total_time = elapsed_time * (total_conversations / total_completed)
                remaining_time = estimated_total_time - elapsed_time
                eta = f"{remaining_time/60:.1f}min"
            else:
                eta = "unknown"
            
            batches_in_group = len(batch_group)
            avg_group_time = group_time / batches_in_group if batches_in_group > 0 else 0

            self.logger.info(f"Group {group_idx}/{len(batch_groups)}: {total_completed}/{total_conversations} ({progress:.1f}%) | "
                           f"Group time: {group_time:.1f}s ({avg_group_time:.1f}s/batch) | ETA: {eta}")
            # Let's save checkpoint periodically
            if group_idx % 5 == 0:
                self._save_checkpoint(checkpoint_file, all_results)
                self.logger.info(f"Checkpoint saved: {len(all_results)} conversations")

            # Adaptive delay based on group processing time
            if group_time < 2.0:
                await asyncio.sleep(0.1)   
            elif group_time > 30.0:
                await asyncio.sleep(1.0)   
            else:
                await asyncio.sleep(0.2)  
        
        self.save_jsonl(all_results, output_file)
        

        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        total_time = time.time() - start_time
        success_rate = len(all_results) / total_conversations * 100 if total_conversations > 0 else 0
        avg_time_per_conv = total_time / total_conversations if total_conversations > 0 else 0
        avg_api_time = self.total_api_time / max(self.total_api_calls, 1)
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1) * 100

        self.logger.info("CONCURRENT INSTRUCTION TRANSLATION COMPLETED")
        self.logger.info("PERFORMANCE METRICS:")
        self.logger.info(f"   • Total: {total_conversations} conversations")
        self.logger.info(f"   • Successful: {len(all_results)} ({success_rate:.1f}%)")
        self.logger.info(f"   • Failed batches: {len(self.failed_batches)}")
        self.logger.info(f"   • Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        self.logger.info(f"   • Avg time/conv: {avg_time_per_conv:.2f}s")
        self.logger.info(f"   • API calls: {self.total_api_calls}")
        self.logger.info(f"   • Avg API time: {avg_api_time:.2f}s")
        self.logger.info(f"   • Cache hit rate: {cache_hit_rate:.1f}%")
        
        return all_results

    async def close(self):
        self.logger.info("Closing PROCESS...")
        await self.client.close()
        self.logger.info("PROCESS closed")