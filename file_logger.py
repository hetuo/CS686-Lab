#!/usr/bin/python
from logger import logger

class file_logger(logger):

    def __init__(self, log_level, file_name = 'file_log.txt'):
	self.__log_level__ = log_level
	self.__file_name__ = file_name	

    def log(self, log_level, message):
	if log_level <= self.__log_level__:
	  log_file = open(self.__file_name__, "a+")
	  log_file.write(str(log_level) + ":" + message + '\n')
          log_file.close()
	return
	
