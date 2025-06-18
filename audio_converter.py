#!/usr/bin/env python3

import os
import logging
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Set, Dict, Any
from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import signal
import sys

try:
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError
except ImportError:
    raise ImportError("pydub is required. Install with: pip install pydub")

class AudioConverterError(Exception):
    pass

class UnsupportedFormatError(AudioConverterError):
    pass

class FileProcessingError(AudioConverterError):
    pass

class SecurityError(AudioConverterError):
    pass

class AudioConverter:
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    SUPPORTED_INPUT_FORMATS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.wma'}
    SUPPORTED_OUTPUT_FORMATS = {'.flac', '.wav', '.mp3', '.m4a', '.ogg'}
    CONVERSION_TIMEOUT = 300  # 5 minutes
    
    def __init__(self, log_level: int = logging.INFO):
        self._setup_logging(log_level)
        self._lock = threading.RLock()
        self._active_conversions = 0
        self._max_concurrent = 4
        
    def _setup_logging(self, level: int) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _validate_file_security(self, file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found")
            
        if not file_path.is_file():
            raise SecurityError("Path is not a regular file")
            
        if file_path.stat().st_size > self.MAX_FILE_SIZE:
            raise SecurityError(f"File exceeds maximum size limit")
            
        try:
            file_path.resolve(strict=True)
        except (OSError, RuntimeError):
            raise SecurityError("Invalid file path")
            
        if any(part.startswith('.') and part != file_path.name for part in file_path.parts):
            raise SecurityError("Hidden directories in path not allowed")
    
    def _get_file_hash(self, file_path: Path) -> str:
        hash_obj = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
        except IOError as e:
            raise FileProcessingError(f"Cannot read file for hashing: {e}")
        return hash_obj.hexdigest()[:16]
    
    def _sanitize_output_path(self, input_path: Path, output_format: str) -> Path:
        stem = "".join(c for c in input_path.stem if c.isalnum() or c in ('-', '_'))
        if not stem:
            stem = f"converted_{self._get_file_hash(input_path)}"
        
        output_name = f"{stem}{output_format}"
        return input_path.parent / output_name
    
    @contextmanager
    def _temp_output_file(self, final_path: Path):
        temp_dir = final_path.parent
        with tempfile.NamedTemporaryFile(
            suffix=final_path.suffix,
            dir=temp_dir,
            delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            yield temp_path
            if temp_path.exists():
                temp_path.replace(final_path)
        except Exception:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise
    
    def _load_audio_segment(self, file_path: Path, timeout: int = 30) -> AudioSegment:
        def load_audio():
            try:
                return AudioSegment.from_file(str(file_path))
            except CouldntDecodeError as e:
                raise FileProcessingError(f"Cannot decode audio file: {e}")
            except Exception as e:
                raise FileProcessingError(f"Error loading audio: {e}")
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(load_audio)
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                raise FileProcessingError("Audio loading timed out")
    
    def _export_audio(self, audio: AudioSegment, output_path: Path, 
                     output_format: str, **kwargs) -> None:
        export_params = {
            'format': output_format.lstrip('.'),
            'parameters': []
        }
        
        format_configs = {
            '.flac': {'parameters': ['-compression_level', '8']},
            '.mp3': {'bitrate': '320k'},
            '.wav': {},
            '.m4a': {'bitrate': '256k'},
            '.ogg': {'parameters': ['-q:a', '6']}
        }
        
        if output_format in format_configs:
            export_params.update(format_configs[output_format])
        
        export_params.update(kwargs)
        
        try:
            audio.export(str(output_path), **export_params)
        except Exception as e:
            raise FileProcessingError(f"Export failed: {e}")
    
    def convert_audio(self, input_file: str, output_format: str = '.flac', 
                     **export_kwargs) -> Optional[str]:
        
        with self._lock:
            if self._active_conversions >= self._max_concurrent:
                raise AudioConverterError("Maximum concurrent conversions reached")
            self._active_conversions += 1
        
        try:
            return self._convert_audio_internal(input_file, output_format, **export_kwargs)
        finally:
            with self._lock:
                self._active_conversions -= 1
    
    def _convert_audio_internal(self, input_file: str, output_format: str, 
                               **export_kwargs) -> Optional[str]:
        
        try:
            input_path = Path(input_file).resolve()
        except (OSError, RuntimeError) as e:
            self.logger.error(f"Invalid input path: {e}")
            return None
        
        try:
            self._validate_file_security(input_path)
        except (SecurityError, FileNotFoundError) as e:
            self.logger.error(f"Security validation failed: {e}")
            return None
        
        input_format = input_path.suffix.lower()
        output_format = output_format.lower()
        
        if input_format not in self.SUPPORTED_INPUT_FORMATS:
            self.logger.error(f"Unsupported input format: {input_format}")
            return None
            
        if output_format not in self.SUPPORTED_OUTPUT_FORMATS:
            self.logger.error(f"Unsupported output format: {output_format}")
            return None
        
        if input_format == output_format:
            self.logger.info(f"File already in target format: {output_format}")
            return str(input_path)
        
        output_path = self._sanitize_output_path(input_path, output_format)
        
        if output_path.exists():
            self.logger.warning(f"Output file already exists: {output_path}")
            return str(output_path)
        
        self.logger.info(f"Converting {input_path} to {output_format}")
        
        try:
            with self._temp_output_file(output_path) as temp_path:
                def conversion_task():
                    audio = self._load_audio_segment(input_path)
                    self._export_audio(audio, temp_path, output_format, **export_kwargs)
                
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(conversion_task)
                    future.result(timeout=self.CONVERSION_TIMEOUT)
            
            self.logger.info(f"Conversion completed: {output_path}")
            return str(output_path)
            
        except (FileProcessingError, TimeoutError) as e:
            self.logger.error(f"Conversion failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during conversion: {e}")
            return None
    
    def batch_convert(self, input_files: list, output_format: str = '.flac', 
                     **export_kwargs) -> Dict[str, Optional[str]]:
        results = {}
        
        def convert_single(input_file):
            return input_file, self.convert_audio(input_file, output_format, **export_kwargs)
        
        with ThreadPoolExecutor(max_workers=min(self._max_concurrent, len(input_files))) as executor:
            futures = [executor.submit(convert_single, f) for f in input_files]
            
            for future in futures:
                try:
                    input_file, result = future.result(timeout=self.CONVERSION_TIMEOUT + 30)
                    results[input_file] = result
                except Exception as e:
                    self.logger.error(f"Batch conversion error: {e}")
        
        return results

def signal_handler(signum, frame):
    logging.getLogger().info(f"Received signal {signum}, shutting down gracefully")
    sys.exit(0)

def main():
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    converter = AudioConverter()
    
    input_audio = "path/to/your/audio.mp3"
    
    try:
        converted_audio = converter.convert_audio(input_audio, '.flac')
        
        if converted_audio:
            converter.logger.info(f"Audio ready for processing: {converted_audio}")
        else:
            converter.logger.error("Audio conversion failed")
            sys.exit(1)
            
    except Exception as e:
        converter.logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
