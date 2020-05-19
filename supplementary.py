#!/usr/bin/env python
import math
import os
import re
import scipy.io as sio
from scipy import interpolate as interp
import numpy as np
import string

import vocaltractcontrol

def list_intersection(a, b):
    """Gives the intersection of two given lists.
    Input: a and b - any two lists
    Output: a list whose elements belong both to a and b
    """
    return list(set(a) & set(b))


def format(num, padnum, padfill="0"):
    res = str(num)
    return padfill*(padnum-len(res)) + res


def get_missing_elements(partial, whole):
    """Returns a list of elements in whole that are not present in partial.
    Input: partial and whole - any two lists.
    Output: a list with elements from whole which are not in partial.
    """
    res = [el for el in whole if el not in partial]
    return res

def readjust_target_times(moments, t0, critarts, vtract, mode, slug):
    """Returns a vector of readjusted temporal target vectors for each articulator parameter,
        e.g., tcommon = 2400, but the lips should reach their target position at
        2250, the tongue at 2300 and the jaw at 2480 => a vector
        [2250, 2250, 2250, 2400, 2400, 2400, 2480, 2480...] of size vtract.totalparams"""
    if mode != "SHIFT":
        res = [np.empty(vtract.totalparams) for moment in np.nditer(moments)]
        for index, moment in np.ndenumerate(moments):
            res[int(index[0])].fill(moment)
        return np.array(res)
    readjtcommvecseq = np.empty([moments.shape[0], vtract.totalparams])
    prevtargettimevec = np.empty([vtract.totalparams])
    prevtargettimevec.fill(t0)      # = [t0]*vtract.totalparams         # LEFT HERE
    critartsseq = [vtract.parameter_numbers(currcritarts) \
                            for currcritarts in critarts]
    for tcommidx, (tcommon, currcritarts) in enumerate(zip(moments, critartsseq)):
        nexttargettime = moments[tcommidx+1] if tcommon != moments[-1] else moments[-1]
        readjtcommvect = [tcommon]*vtract.totalparams
        # currcritarts is a list of parameter numbers that correspond to the critical articulators
        for art in currcritarts:
            if prevtargettimevec[art] + 20 < readjtcommvect[art]:
                readjtcommvect[art] = prevtargettimevec[art] + 20
            elif readjtcommvect[art] - prevtargettimevec[art] > 2:
                readjtcommvect[art] = int((prevtargettimevec[art] + readjtcommvect[art])*0.5)
        # At this point, readjtcommvect[art] = prevtargettimevec[art] + (at least 1 at most 20) for every critical articulator
        for art in range(vtract.totalparams):
            corresparticulator = vtract.which[art]
            if corresparticulator==list(corresparticulator):
                corresparticulator = corresparticulator[0]
            speed = vtract.articspeed[corresparticulator]
            readjtcommvect[art] = min(int(prevtargettimevec[art] + (readjtcommvect[art] - prevtargettimevec[art])/speed), nexttargettime)
        # After this section, all articulatory targets are shifted according to their speed
        for art in range(vtract.totalparams):
            if readjtcommvect[art] == prevtargettimevec[art]:
                readjtcommvect[art] = prevtargettimevec[art] + 1
        readjtcommvecseq.append(readjtcommvect)
        prevtargettimevec = readjtcommvect
    if not os.path.isdir("EvalCorpus/Check_new_timings/"):
        os.makedirs("EvalCorpus/Check_new_timings/")
    with open("EvalCorpus/Check_new_timings/checknewtimings_{}_{}.txt".format(slug, mode), "w") as res:
        lines = list()
        sections = [list() for _ in range(vtract.totalparams)]
        for timeslice in readjtcommvecseq:
            for art in range(vtract.totalparams):
                sections[art].append(timeslice[art])
        for section in sections:
            sectiontext = [str(int(el)) for el in section]
            sectiontext = [str(el)+" "*(5-len(el)) for el in sectiontext]
            lines.append(" ".join(sectiontext))
        res.write("\n".join(lines))
    return readjtcommvecseq


def euclidean_distance(a, b):
    """Calculates the Euclidean distance between two given vectors.
    The vectors should be of the same length.
    """
    if len(a) != len(b):
        return -1
    val = math.sqrt(sum([(ai - bi)*(ai - bi) for ai, bi in zip(a, b)]))
    return float("{0:.2f}".format(val))


def estimate_from_corner_vowels(anticipcorner, coeffs):
    """Estimates the vocal tract configuration for a consonant anticipating a
        particular vowel by using the representation of this vowel as a
        combination of corner vowels:
        [vowel] represented as [u] + s ([i] - [u]) + q ([a] - [u])
    Input:
        anticipcorner = [consu:list of float, consi:list of float,
                        consa:list of float] - articulatory vectors for the
            consonant in question anticipating [u], [i], [a] respectively
        coeffs = [s:float, q:float]: coefficients from the representation above
            coordinates of the vowel (that is to be anticipated) in the "u-i-a"
            subspace
    Output:
        estimation: list of float - an articulatory vector for the consonant in
            question, anticipating the vowel whose local coordinates are s and q
    """
    consu, consi, consa = anticipcorner
    s, q = coeffs
    return [float("{0:.2f}".format(uk + s*(ik - uk) + q*(ak - uk))) for uk, ik, ak \
                                                            in zip(consu, consi, consa)]


def interpolate(targets, moments, toestimate, t0, phindexing, tdecipher, critarts,
                vocaltract, slug, mode="COMPLEX"):
    """Interpolates between a given sequence of targets by cosine interpolation
    Input:
        targets: list of lists of float. All elements have to be of the same
            length. These vectors are the target ones for interpolation.
        moments: list of integers: at which moments of time [in ms] the targets
            are to be achieved, respectively.
        toestimate: an integer or a list of integers: at which moment(s) of
            time [in ms] to provide estimations. Elements should be in correct
            temporal order, from the past to the future.
        phindexing: list of tuples (q, n, k), where all elements are
                integers and indicate the phoneme in production in the
                following way:
                    Utterance.synts[q].sylls[n].constituents[k]
                If currently nothing is being produced, the tuple is replaced
                by "#".
                So, gridaddresses may look like
                ["#", "#", (0,0,0), (0,0,1), (0,0,2), (0,1,0), "#", "#"...]
        tdecipher: list of strings. The encoded commentary on what phoneme
            production stage each moment of time in toestimate corresponds to.
            See the format in the function temporal_grid in the Utterance
            class. Currently it is a dummy argument.
        critarts: list of lists of strings. Each list in the list is a list of
            articulators that are critical for producing the current phoneme.
            critarts is a dummy argument unless the mode variable is set to
            "COMPLEX".
        vocaltract: an instance of the VocalTract class to be able to relate
            to the . Expected values: "LIN" to join the target vectors by
            linear segments, and "COS" or "COMPLEX" to perform cosine
             articulators.
mode: string       interpolation. By default it is "COMPLEX".
    Output:
        list of lists of float. Every element is an estimation at the
            corresponding moment of time in toestimate.
    """
    """
    targets = [list(target.vector) for target in targets]
    moments = list(moments)
    if toestimate != list(toestimate):
        toestimate = [toestimate]
    readjtcommvecseq = readjust_target_times(moments, t0, critarts, vocaltract, mode, slug)
    estimation = [list() for _ in toestimate]
    k = 0 # t in toestimate is in segment [moments[k], moments[k+1])
    prevt, nextt = readjtcommvecseq[0], readjtcommvecseq[1]
    for m, t in enumerate(toestimate):
        if (t >= moments[k+1] and t != toestimate[-1]):
            # k+1 = len() - 1 => k = len() - 2
            k += 1
            prevt = readjtcommvecseq[k]
            nextt = readjtcommvecseq[k+1]
        for num, (parprev, parcurr) in enumerate(zip(targets[k], targets[k+1])):
            coscomponent =  1.0*(t - prevt[num])/(nextt[num] - prevt[num])
            lincomponent = 1.0*(nextt[num] - prevt[num])
            if mode == "LIN":
                val = ((parcurr-parprev)*t + parprev*nextt[num] - \
                        parcurr*prevt[num])/lincomponent
            elif (mode == "COS") or (mode == "COMPLEX" and t < nextt[num]):
                val = 0.5*(parprev + parcurr + \
                        (parprev - parcurr)*math.cos(math.pi*coscomponent))
            else:
                val = parcurr
            estimation[m].append(float("{0:.2f}".format(val)))
    """
    targets = np.array([list(target.vector) for target in targets])
    moments = np.array(list(moments))
    toestimate = np.array(list(toestimate))
    readjtcommvecseq = readjust_target_times(moments, t0, critarts, vocaltract, mode, slug) 
    splines = [interp.PchipInterpolator(readjtcommvecseq[:,k], targets[:,k]) for k in range(vocaltract.totalparams)]
    estimation = np.transpose(np.array([spl(toestimate) for spl in splines])).tolist()
    # readjtcommvecseq[:,x] is a particular articulator, readjtcommvecseq[:,x] is a particular moment
    # t = np.array([tpt for tpt, _ in allknots])
    # lch = np.array([lchpt for _, lchpt in allknots])
    # spl = interp.PchipInterpolator(t, lch)
    # allt = np.arange(allknots[0][0], allknots[-1][0])
    # lchestim = spl(allt)
    return estimation


def strip_but_alphanum(s):
    pattern = re.compile('[\W_]+')
    return pattern.sub('', s)


def align(decodinginstructions, afs):
    # decodinginstructions = [(time, phoneme, decoding), ...]
    # afs = [(aftime, afname)]
    # res = [((time, phoneme, decoding), (aftime, afname)), ...]
    afdict = dict()
    res = list()
    for aftime, afname in afs:
        afdict[aftime] = afname
    for time, phoneme, decoding in decodinginstructions:
        if afdict.has_key(time):
            res.append(((time, phoneme, decoding), (time, afdict[time])))
        else:
            res.append(((time, phoneme, decoding), (time, res[-1][1][1])))
    return res


def undo_the_prev_test(loc, trash, overallareaext=".xa", velumareaext=".xav", afext='.xainterm', resultext=".xax"):
    # Delete the intermediate files (afext) and resultext files
    # Move the .xa and .xav files from trash back to loc
    if (not os.path.exists(trash)) or (not os.path.exists(loc)):
        return
    for f in os.listdir(loc):
        if any(f.endswith(ending) for ending in [overallareaext, velumareaext, afext, resultext]):
            os.remove(os.path.join(loc, f))
    for f in os.listdir(trash):
        if any(f.endswith(ending) for ending in [overallareaext, velumareaext]):
            os.rename(os.path.join(trash, f), os.path.join(loc, f))
        elif any(f.endswith(ending) for ending in [afext, resultext]):
            os.remove(os.path.join(trash, f))
    
    

def correct_xax_files(xaxlocation, intermlocation, uttfile, affile, t0, loc2glob, afext='.xainterm', resultext=".xax"):
    tubeidentification = {"Bilabial": list(range(37, 40)), "Dental": list(range(36, 39)), "Palato-alveolar": list(range(34, 37)), "Palatal": list(range(32, 36)), "Velar": list(range(17, 34)), "Uvular": list(range(21, 30))} 
    placesofarticulation = list(tubeidentification.keys())
    trashfolder = os.path.join(intermlocation, "Trash")
    ensure_dir(trashfolder)
    with open(intermlocation + "afcorrection.log", "a") as logf:
        samemessagedetector = None
        with open(uttfile, 'r') as uttdecodf:
            # [(time, phoneme, decoding) line after line]
            decodinginstructions = [(instr[:instr.find('.')], instr[instr.find('[')+1:instr.find(']')], instr[instr.rfind(' ')+1:]) for instr in uttdecodf.readlines()[1:] if int(instr[:instr.find('.')]) >= t0]
            with open (affile, 'r') as affunf:
                # Actually every area function may be used multiple times, and there is no guarantee that its corrections should be the same at all instances.
                affilelines = affunf.readlines()
                times = [time[:time.find(".")] for time in affilelines[3::4]]
                filenames = [string.rstrip(name) for name in affilelines[4::4]]
                afs = zip(times, filenames)
                aligned = align(decodinginstructions, afs)
                for ((t, ph, decod), (aft, af)) in aligned:
                    if t != aft:
                        errormessage = "The instructions are inconsistent at t = %s, aft = %s." % (t, aft)
                        print errormessage
                        logf.write(errormessage + "\n")
                phonemes = [vocaltractcontrol.Phoneme(ph) if "#" not in ph else "#" for _, ph, _ in decodinginstructions]
                instructions = [(int(t), ph, af, decod, phonemes[k]) for (k, ((t, ph, decod), (aft, af))) in enumerate(aligned)]
                for k, (t, phname, af, decod, ph) in enumerate(instructions):
                    if "#" not in phname:
                        # CORRECT THE PATH ACCORDING TO xaxlocation AND loc2glob
                        localaf = os.path.join(os.path.dirname(uttfile), af).replace("\\", "/")
                        if loc2glob.has_key(localaf):
                            translatedaf = loc2glob[localaf]
                        else:
                            print "Problem with " + localaf + ": it cannot be translated into an area function for the corpus."
                            continue
                        # f = os.path.join(intermlocation, af[:-len(resultext)] + afext)
                        f = translatedaf[:-len(resultext)] + afext
                        if not os.path.isfile(f):
                            if samemessagedetector != translatedaf:
                                logf.write("\"%s\" does not need to be processed because it has already been processed.\n" % translatedaf)
                            samemessagedetector = translatedaf
                            continue
                        decodnomovement = strip_but_alphanum(decod)
                        logf.write(" -- ".join([str(el) for el in [t, phname, localaf, translatedaf, decod, decodnomovement, ph.artfeatures if ph != "#" else "#", ph.phclass if ph != "#" else "#"]]) + "\n")
                        with open(f) as curraff:
                            # print "Processing " + localaf[:-len(resultext)] + afext + " (" + f + ")..."
                            afdirectives = curraff.readlines()
                            numoftubes = int(afdirectives[1])
                            correctedxax = [afdirectives[0].replace(afext, resultext), afdirectives[1]]
                            correctedvelumopening = float(afdirectives[2][afdirectives[2].rfind(" ")+1:])
                            if decodnomovement != "N" and phname[-1] != "~":    # Not a nasal => velum opening = 0
                                # Warning: hidden usage of a way to identify nasal vowels by their names ("o~", etc.)
                                logf.write("This is not a nasal sound, so the opening %f needs to be corrected to 0.\n" % correctedvelumopening)
                                correctedvelumopening = 0.0
                            elif correctedvelumopening < 0.5:                   # Nasal with a very small opening => 0.5  
                                logf.write("This is a nasal sound, so the opening %f needs to be corrected to 0.5.\n" %correctedvelumopening)
                                correctedvelumopening = 0.5
                            correctedxax.append(string.rstrip(afdirectives[2][:afdirectives[2].rfind(" ")+1]) + " " + str(correctedvelumopening) + "\n")
                            # velumopening = float(afdirectives[2][afdirectives[2].rfind(" ")+1:])
                            tubes = [(line[:line.find(" ")], float(line[line.find(" ")+1:line.rfind(" ")]), string.rstrip(line[line.rfind(" ")+1:])) for line in afdirectives[3:3+numoftubes]]
                            smallestval, smallestwhere = tubes[0][1], 0
                            for (k, (x, y, z)) in enumerate(tubes):
                                if y < smallestval:
                                    smallestval = y
                                    smallestwhere = k
                                correctedy = y
                                if ph.phclass == "V":
                                    if correctedy < 0.5:
                                        vowelcorrection = None
                                        cval, mcval, moval, oval = 0.25, 0.3, 0.35, 0.4
                                        if decodnomovement == "c" and correctedy < cval:
                                            vowelcorrection = cval
                                        elif decodnomovement == "mc" and correctedy < mcval:
                                            vowelcorrection = mcval
                                        elif decodnomovement == "mo" and correctedy < moval:
                                            vowelcorrection = moval
                                        elif decodnomovement == "o" and correctedy < oval:
                                            vowelcorrection = oval
                                        if vowelcorrection and correctedy != vowelcorrection:
                                            logf.write("%s: The opening at tube #%d is too small for a vowel /%s/, %f. It is corrected into %f.\n" %(af, k, phname, correctedy, vowelcorrection))
                                            correctedy = vowelcorrection
                                        else:
                                            logf.write("%s: The opening at tube #%d is rather small for a vowel /%s/, %f. It is not corrected, though.\n" %(af, k, phname, correctedy))
                                    # If there is an upper bound for tube volume for vowels, but the block here
                                else: # ph.phclass == "C"
                                    if decodnomovement == "F" and correctedy < 0.1: # This is a constriction. You can use k to identify where it is
                                        logf.write("%s: The opening at tube #%d is too small for a fricative /%s/, %f. It is corrected into 0.1.\n" %(af, k, phname, correctedy))
                                        correctedy = 0.1
                                    elif decodnomovement in ["S", "N"] and correctedy < 0.05:
                                        stopplace = None
                                        for poa in placesofarticulation:
                                            if poa in ph.artfeatures and k in tubeidentification[poa]:
                                                stopplace = poa
                                        if stopplace != None:    
                                            logf.write("%s: Tube #%d seems to the place of articulation for the current stop /%s/ (%s), and %f is corrected into 0.\n" %(af, k, phname, stopplace, correctedy))
                                            correctedy = 0
                                correctedxax.append(x + " " + str(correctedy) + " " + z + "\n")
                            logf.write("The narrowest constriction of %f is achieved at the tube #%d.\n" % (smallestval, smallestwhere))
                        # fname = af[:-len(resultext)] + afext
                        # f = os.path.join(intermlocation, fname)
                        try:
                            os.rename(f, os.path.join(trashfolder, os.path.basename(f)))
                        except:
                            print "Storing the original area function didn't work for " + f
                        with open(translatedaf, 'w+') as areafunction:
                            areafunction.write("".join(correctedxax))


def correct_xa_files(location, t0, overallareaext=".xa", velumareaext=".xav", \
                     velumopmarker="Velum opening in cm", resultext=".xainterm"):
    """Transforms the vocal tract and velum area function files into the input
    format of the synthesizer.
    Input:
        location: string - path to the folder that will be walked recursively.
        overallareaext: string - the extension of the area function files. By default,
            ".xa".
        velumareaext: string - the extension of the velum area function files.
            It can be also set None to avoid subtracting from the overall area
            functions the velum ones.
        resultext: string - the extension of the files for the synthesiser.
    Output:
        None.
    """	
    numoftubes = 0
    trashfolder = os.path.join(location,"Trash")
    ensure_dir(trashfolder)
    for f in [os.path.join(dp, f) for dp, dn, fn in os.walk(location) \
                                    for f in fn if f.endswith(overallareaext)]:
        try:
            with open(f, "r") as areafunction:
                # aflines = [area for area in list(areafunction) if area != "\n"][2:]
                aflines = [area for area in list(areafunction) if area != "\n"]
                numoftubes = len(aflines)
                areafunction.close()
                if velumareaext != None:
                    tubes = [float(area.split(" ")[0]) for area in aflines]
                    overallareas = [float(area.split(" ")[1]) for area in aflines]
                    velumareas = [0.0]*numoftubes
                    fvel = f[:f.rfind(".")]+velumareaext
                    with open(fvel, "r") as velumareafunction:
                        #vaflines = [area for area in list(velumareafunction) \
                        #            if area != "\n"][2:]
                        vaflines = [area for area in list(velumareafunction) \
                                    if area != "\n"]
                        try:
                            if vaflines[0].startswith(velumopmarker):
                                velopening = float(vaflines[0][len(velumopmarker)+1:])
                        except:
                            return None
                        velumareas = [float(area.split(" ")[1]) for area in vaflines[1:]]
                # differences = [m-s for m, s in zip(overallareas, velumareas)]
                resf = f[:f.rfind(".")]+resultext
                #af = "[{}]\n{}\n{}: {}\n".format(resf[resf.rfind(os.pathsep)+1:],
                #                                str(numoftubes), velumopmarker,
                #                                velopening)
                af = "[{}]\n{}\n{}: {}\n".format(resf[resf.rfind("A"):],
                                                str(numoftubes), velumopmarker,
                                                velopening)
                # for tube, area in zip(tubes, differences):
                for tube, totalarea, velumarea in zip(tubes, overallareas, velumareas):
                    af += "{} {} {}\n".format(tube, totalarea, velumarea)
                with open(resf, "w") as areafunctionfull:
                    areafunctionfull.write(af)
                    tmp = f[:f.rfind("/")]
                    # whereto = tmp if tmp.find("/") == -1 else tmp[tmp.rfind("/")+1:]
                    # whereto = os.path.join(trashfolder, whereto)
                    # if (not os.path.exists(whereto)) or not os.path.isdir(whereto):
                    #     os.makedirs(whereto)
                    os.rename(f, os.path.join(trashfolder, f[f.rfind("/")+1:]))
                    os.rename(fvel, os.path.join(trashfolder, fvel[fvel.rfind("/")+1:]))
                    # print "Files \"{}\" and \"{}\" have been processed.".format(f, fvel)
                    areafunctionfull.close()
        except:
            print "Either AFs {} have been processed before, or it didn't work for them."



def produce_matlab_list(d, outputf="MatlabList.txt"):
    corpusfolder = d.replace("/", "").replace("\\", "") + "_data"
    datafolders = ["\'" + o[:-5] + "\'" for o in os.listdir(d) if os.path.isdir(os.path.join(d,o)) and os.path.join(d,o) != corpusfolder]
    with open(d+outputf, "w") as f:
        f.write("{" + ", ".join(datafolders) + "}")
        print "The list of directories for Matlab has been stored in " + d+outputf + "."


def produce_xarticul_calls(location, prefix="xarticul ", extension=".xart",
                           outputf="XarticulCalls.txt", period=1, mode="at"):
    """Recursively scans the given directory for scripts and provides
    a string that is a sequence of calls to xarticul to run the scripts of
    the given extension.
    Input:
        location: path to a folder.
        prefix: how to call Xarticul. By default, "xarticul ".
        extension: what is the extension of Xarticul scripts. By default,
            ".xart".
        outputf: where in location to store the output file. If outputf is set
            as None, the sequence of calls to Xarticul will simply be printed
            in the console.
    """
    scripts = [f for dp, dn, fn in os.walk(location)
                    for f in fn
                        if f.endswith(extension)]
    if mode=="at":
        res = prefix + scripts[0] + " &\n"
        minutes = range(period, period*len(scripts), period)
        for min, scr in zip(minutes, scripts[1:]):
            res += "echo xarticul {} | at now + {} min;\n".format(scr, min)
        res = res[:-3]
    else:
        scriptgroups = [scripts[k:k+period] for k in range(0, len(scripts), period)]
        res = "\n\n".join([prefix + (" & " + prefix).join(group) for group in scriptgroups])
    if outputf != None:
        with open(location+outputf, "w") as calls:
            calls.write(res)
            print "Xarticul calls have been stored in " + location+outputf + "."
            return None
    print prefix + ("; "+prefix).join(scripts)

                
def produce_list_for_synthesis(location, ending="_data"):
    """Transforms the area function files into the input format of the synthesizer.
    """	
    qualifying = [d[:d.rfind(ending[0])] for dp, dn, fn in os.walk(location) for d in dn if d.endswith(ending)]
    with open(location+"utterances.txt", "w") as uttlist:
        uttlist.write("'"+"', '".join(qualifying)+"'")

def resize(arr, maxv):
    currmaxv = max(el[1] for el in arr)
    currminv = min(el[1] for el in arr)
    currdiff = currmaxv - currminv
    diff = maxv - currminv
    def rel_pos(val):
        return (val-currminv)/currdiff
    newarr = list()
    for t, val in arr:
        newarr.append((t, currminv+rel_pos(val)*diff))
    return newarr

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
def delete_if_exist(files):
    for f in files:
        if os.path.exists(f):
            os.remove(f)

def reduce_data(matfile, timearg, vararg, tshift, maxf0=None):
    """Loads the given .mat file and produces the instructions that are required
        for copy synthesis.
    Input:
        matfile: string. The address of the .mat file to be loaded.
        timearg: string. The name of the time variable encoded in the .mat file.
        vararg: string. The name of the variable that we want to load from the file.
        tshift: int. The time shift between the values for synthesis.
    Output:
        extr: list of tuples that contain two integers: time and value.
    """
    mat = sio.loadmat(matfile)
    # Going ms instead of s and shifting the time axis:
    time = [tshift+int(1000*el) for el in mat[timearg][0]]
    # Warning: precision is not a parameter. It is set to 3.
    if len(mat[vararg]) > 1:
        var = [float("{0:.3f}".format(el)) for part in mat[vararg] for el in part]
    else:
        var = [float("{0:.3f}".format(el)) for el in mat[vararg][0]]
    for k, val in enumerate(var):
        if not math.isnan(val):
            lastmeaningful = val
        else:
            var[k] = lastmeaningful
            print "Nan values found at time " + str(time[k]) + "..."
    sample = zip(time, var)
    # If we observe different values at the same millisecond, we average them:
    flt = list() # flt: "flatten by time"
    for t in sorted(list(set(time))):
        relevvals = [observ[1] for observ in sample if observ[0]==t]
        avg = float("{0:.3f}".format(sum(relevvals)/float(len(relevvals))))
        flt.append((t, avg))
    # If we observe the same value over consecutive milliseconds,
    # we take the mean time stamp (or time stamps):
    flv = list() # flv: "flatten by value"
    kstrt = 0 # kstrt: "k start", the index k at which the current value started
    for (k, (t, v)), nxv in zip(enumerate(flt), [x[1] for x in flt[1:]]+[-1]):
        if v != nxv: # nxv: "next value"
            rft = flt[kstrt][0] # rft: "reference time"
            kstrt = k + 1
            # Warning: an uncontrolled parameter of 40 ms that regulates
            # whether we create one or two datapoints in the time period:
            if t - rft < 40:
                flv.append((int(0.5*(t+rft)), v))
            else:
                # Warning: an uncontrolled parameter of 0.25 regulating at what
                # moments of time we put the two datapoints:
                flv.extend([(int(0.25*t+0.75*rft), v), (int(0.75*t+0.25*rft), v)])
    if maxf0:
        flv = [(t, v) for (t, v) in flv if t>1990]
        extr = resize(flv, maxf0)
    else:
        extr = flv
    return extr

        
# correct_xa_files("../Resynthesis/AF/")
# produce_xarticul_calls("EvaluationNew/")