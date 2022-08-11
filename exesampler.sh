#! usr/bin/bash

# Okay
#python sampler.py -s  0 -t GRB160530667
#python sampler.py -s 37 -t GRB120323507
python sampler.py -s 57                 # GRB090304216
#python sampler.py -s 49 -t GRB121122885
#python sampler.py -s 97 -t GRB141030746
#python sampler.py -s 87 -t GRB150320462
#python sampler.py -s 0                  # GRB170131969
#python sampler.py -s 17 -t GRB170403583
#python sampler.py -s 37 -t GRB170921168
#python sampler.py -s 37 -t GRB171010792
#python sampler.py -s 37 -t GRB180313978

# Problems
#python sampler.py -s 42                 # GRB090519881
#python sampler.py -s 58 -t GRB120817168 -nai=0 # Only BGO. NaI fail backfitter
#python sampler.py -s  7 -t GRB131126163 # Only BGO, backfitter 1
#python sampler.py -s 47 -t GRB140818781 # Tutto negativo ->0
#python sampler.py -s  7                 # GRB180111815
