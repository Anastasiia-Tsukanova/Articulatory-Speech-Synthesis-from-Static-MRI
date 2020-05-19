#!/usr/bin/env python

"""
Authors:         Yves LAPRIE, Anastasiia TSUKANOVA
            
Articulatory Speech Synthesis
"""

from vocaltractcontrol import Utterance
from vocaltractcontrol import Corpus
from vocaltractcontrol import scan_directory_for_phonemes as scan_db
from vocaltractcontrol import database_expand as db_expand
from vocaltractcontrol import evaluate_projections as eval_projections
# from vocaltractcontrol import connect_to_existing_corpus
from supplementary import produce_list_for_synthesis as produce_call_list, correct_xa_files, produce_xarticul_calls, produce_matlab_list, undo_the_prev_test
import visualisationInterface

currworkingfolder = "copysynthesis/"
# currcorpus = "newcorpus"

copymatfrom1folder = "EPGGmodelling/First/"
# copymatfrom2folder = "EPGGmodelling/Second/"
# copygridfromfolder = "ilzappepasmal_imit/"

# db_expand()
# eval_projections()

vcv = ["/a:-_\\pa:", "/a:-_\\ba:", "/a:-_\\ta:",
        "/a:-_\\da:", "/a:-_\\ka:", "/a:-_\\ga:",
        "/a:-_\\fa:", "/a:-_\\va:", "/a:-_\\sa:",
        "/a:-_\\za:", "/a:-_\\{ch}a:",
        "/a:-_\\{zh}a:", "/a:-_\\la:", "/a:-_\\ra:",
        "/a:-_\\ma:", "/a:-_\\na:", "/a:-_\\wa:",
        "/a:-_\\ja", "a:-//pu:", "y:-//bi:", "{epsilon~}:-/te:",
        "o:-/dy:", "i:-/ka:", "{oe}:-/k{a~}:", "{o~}:-/ke:", "{deux}:-/g{epsilon}:",
        "\\y:-\\g{o~}:", "{a~}:-/ga:", "i:-/ni:", "i:-/fy:", "u:-/v{epsilon~}:",
        "{o~}:-/se:", "a:-/zi:", "u:-/{ch}{o~}:", "{epsilon}:-/{zh}a:",
        "a:-/l:e:", "i:-/l:y:", "o:-/ri:", "u:-/r{a~}:", "a:-/mi:",
        "{oe}:-/m{o~}:", "o:-/ne:", "{epsilon}:-/ny:", "a:-/wi:"]
        #"a:-\\fa:", "a:-\\va:", "a:-\\sa:", "a:-\\{ch}a:", "a:-\\za:", "a:-\\{zh}a:"]
cvc = ["ba:-\\r{epsilon}", "sa:-\\{zh}{epsilon}", "tu:t", "si:s", "ta:-\\l{epsilon}"]
vccv = ["a:k-\\ra", "ab-\\ri:"]
phrases = ["\\b{o~}/{zh}u:r",
           "\\l{epsilon}\\za\\ba/_{zh}'u:r | s{o~}t{o~}m\\b'e:",
           "ila\\pa\\_ma:l",
           "ka://m'y:",
           "\\{zh}{oe}k/rwa:",
           "{zh}{oe}/s{epsilon}\\k{oe}/s{epsilon}ba/na:l | \\m{epsilon}:{epsilon}: | l{oe}pro/bl{epsilon}m{epsilon}\la",
           "\\es\\'k{oe}\\ty\\p{deux}:\\{deux}:\\{deux}: | pr{a~}:dr{epsilon}{epsilon~}r{a~}//de//vu:"]

undo_the_prev_test(currworkingfolder+currworkingfolder[:-1]+"_data/", currworkingfolder+currworkingfolder[:-1]+"_data/Trash/")
testcorpus = Corpus(currworkingfolder[:-1], vcv + vccv + cvc + phrases) 
testcorpus.record(outputloc=currworkingfolder, coartmode="SPL", visualise=False)
testcorpus.process_AFs()

# produce_xarticul_calls(currworkingfolder, period=6, mode="accumulate")

# produce_matlab_list(currworkingfolder)

# correct_xa_files(currworkingfolder)