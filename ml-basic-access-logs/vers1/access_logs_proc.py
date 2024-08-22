# Process access logs lines
# See log format: https://httpd.apache.org/docs/trunk/mod/mod_log_config.html
import os
import sys
import csv
import datetime
import random
import time
import re
import traceback
import re

def process_line(line, ii):    
    # Use regex to capture access log data
    x1 = re.search(r'^(\d+\.\d+\.\d+\.\d+) (\d+\.\d+\.\d+\.\d+) - ([A-Za-z0-9\-=,_]+) \[(.*)\] \"(.*)\" (\d+) (\d+) \".*\" \"(.*)\" ([0-9]+) (.*)$'
                  , line)
    active_re = None
    if (x1 is not None):
        active_re = x1
    if (active_re is not None):
        url_group = None
        url_val = None
        response_time = 0
        if (active_re is not None and active_re.group(5) is not None):
            url_group = active_re.group(5)
            url_val = url_group.split()[1]
            response_time = active_re.group(9)
        if (url_val is not None):        
            slow_time_check = 8000.0
            # Convert from microseconds to milliseconds    
            response_time_tot_ms = (float(response_time) / 1000.0)
            #print(str(ii) + ",2014-02-27 16:50:00," + str(response_time_tot_ms))
            print("2014-02-27 16:50:00," + str(response_time_tot_ms))

def read_and_parse_access(file):
    print('Parsing file for data analysis ', file)
    with open(file) as infile:
        for i, line in enumerate(infile):
            # print("Processing line number: ", i)
            if (len(line) > 10):
                try:
                    process_line(line, i)    
                except Exception as e:
                    print("Error processing line ", i)    
                    traceback.print_exc()

    # Dont process data
    print("Done processing data file, now collecting data")
    
# Main entry point into program
def main():     
    print('*** Running preprocess *****') 
    print('*** Number of arguments:', len(sys.argv), 'arguments.')
    print('*** Argument List:', str(sys.argv))
    
    # First argument is to path log file
    if len(sys.argv) == 2:    
        read_and_parse_access(sys.argv[1])
    else:
        print("Invalid arguments, enter access log file")
        
if __name__ == '__main__':
    main()                
