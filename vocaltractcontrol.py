#!/usr/bin/env python
import os
import re
import numpy as np
import time
from shutil import copy2
from scipy import interpolate as interp

from supplementary import list_intersection as list_intersect
from supplementary import get_missing_elements
from supplementary import estimate_from_corner_vowels
from supplementary import euclidean_distance as euclid_dist
from supplementary import interpolate
from supplementary import reduce_data as imp
from supplementary import format
from supplementary import correct_xa_files
from supplementary import correct_xax_files
from supplementary import produce_matlab_list
from supplementary import ensure_dir
from supplementary import delete_if_exist

import visualisationInterface

class VocalTract(object):
    """A class for describing the used vocal tract model.
    Language: French
    Model: "Data/VT-Model/antoineIRM.pca"
    Captures: "Data/VT-Model/Contours-to-Build-The-Model/*.[ctr|pgm]"
    Visualisation: "Data/Speech-Synthesis-Database/Images/*.png"
    """
    def __init__(self):
        """Configures the parameters of the vocal tract model.
        Attributes self.nbjaw, self.nbtongue, self.nblips, self.nbepiglottis,
            self.nblarynx, and self.nbvelum (all integers) set the number of
            parameters per each articulator.
        Articulator's name is set at the corresponding position in list
            paramnames (list of strings), from where it is transferred to all
            attributes where it is needed (see below). 
        You can also use the short name which is to be set in parampseuds, a
            list of strings, from where it is transferred to all attributes
            where it is needed (see below).
        Any articulator can be split into named subarticulators in
            self.complexarticulators with the following structure:
            key: name of the articulator to be split (string)
            value: list of tuples,
                    each tuple is of three values:
                    -   full name of the part of the articulator (string)
                    -   its short name (string)
                    -   how many parameters are given to this part.
            Take note that the parts are ordered.
        self.shortnames and self.fullnames are dictionaries for translating
            short names into full names and vice versa.
        self.articulators is a dictionary whose keys are strings that are
            full or short names of articulators or their parts,
            and values are lists of integers: the parameter numbers in an
            articulatory vector that correspond to the particular articulator
            or its part.
        self.fullnarticulators is a copy of self.articulators, but only with
            fully named keys.
        self.superficialorder contains the correct order of superarticulators
            (list of strings, the elements are full names of articulators)
        self.lowerorder (list of strings) contains the correct order of the
            lowest layer of articulators: either simple ones or their subparts
            instead of superarticulators
        self.complexorder is a list whose elements are strings (for the case of
            a simple articulator) or lists of strings (for a complex
            rticulator).
        self.totalparams: integer - how many articulatory parameters encode a
            vocal tract configuration.
        self.nbarticulators: integer - how many articulators are distinguished
            in a vocal tract.
        self.which is a list of length self.totalparams; each element in it
            tells which articulator is responsible for this position, e.g.:
            ["Jaw", "Jaw", "Jaw", "Tongue", ...]
            Complex articulators are handled as lists:
            ["Jaw", ..., "Tongue", ["Lips", "LipsProtrusion"],
                ["Lips", "LipsAperture"], "Epiglottis", ...]
        """
        self.model = "Data/VT-Model/antoineIRMext.pca"
        self.nbjaw = 3
        self.nbtongue = 12
        self.nblips = 3
        self.nbepiglottis = 3
        self.nblarynx = 3
        self.nbvelum = 5
        # Complex articulators: after handling the superficial level
        params = [self.nbjaw, self.nbtongue, self.nblips,
                    self.nbepiglottis, self.nblarynx, self.nbvelum]
        # Now the order of articulators is set:
        # A 3-parameter-long one, a 12-parameter-long one...
        paramnames = ["Jaw", "Tongue", "Lips", "Epiglottis",
                        "Larynx", "Velum"]
        parampseuds = ["J", "T", "Li", "E", "La", "Vel"]
        # Addition on 14/03/2017: the articulators can move at different speeds
        self.articspeed = {"Jaw": 0.75, "Tongue": 1.0, "Lips": 1.5, "Epiglottis": 1.2,
                        "Larynx": 1.0, "Velum": 1.2}
        self.complexarticulators = {"Lips": [("LipsProtrusion", "LiPr", 1),
                                                ("LipsAperture", "LiA", 1),
                                                ("LipsFull", "LiF", 1)]}
        self.subarticulators = list() # A list of names of subarticulators
        #                      (used for easy access from other program units)
        for k, v in self.complexarticulators.iteritems():
            self.subarticulators.extend([part[0] for part in v])
            self.subarticulators.extend([part[1] for part in v])
            n = paramnames.index(k)
            correcttotal = params[n]
            if sum([part[2] for part in v]) != correcttotal:
                print "Your structure of the articulators' model is not correct."
                self.status = "ERROR"
                return
            else:
                self.status = "+"
        self.shortnames = dict() # dictionary from short names to the full ones
        self.fullnames = dict() # dictionary from full names to the short ones
        self.articulators = dict()
        self.fullnarticulators = dict() # Just like self.articulators, but only
        #               with full-name keys.
        self.totalparams = sum(params)
        self.nbarticulators = len(params)
        self.which = [None]*self.totalparams
        previous = 0
        for limit, name, shortname in zip(params, paramnames, parampseuds):
            articulatorspan = range(previous, previous+limit)
            self.articulators[name] = articulatorspan
            self.fullnarticulators[name] = self.articulators[name]
            self.articulators[shortname] = self.articulators[name]
            self.shortnames[shortname] = name
            self.fullnames[name] = shortname
            if not self.complexarticulators.has_key(name):
                previous += limit
                for par in articulatorspan:
                    self.which[par] = name
            else:
                # This is a complex articulator.
                parts = self.complexarticulators[name]
                for partfullname, partshortname, partlimit in parts:
                    partspan = range(previous, previous+partlimit)
                    self.articulators[partfullname] = partspan
                    self.fullnarticulators[partfullname] = self.articulators[partfullname]
                    self.articulators[partshortname] = self.articulators[partfullname]
                    self.shortnames[partshortname] = partfullname
                    self.fullnames[partfullname] = partshortname
                    previous += partlimit
                    for param in partspan:
                        self.which[param] = [name, partfullname]
        self.superficialorder = paramnames
        # Superficial: ["Jaw", "Tongue", "Lips", "Epiglottis", ...]
        self.lowerorder = list()
        # Lower order: ["Jaw", "Tongue", "LipsAperture", "LipsProtrusion", ...]
        self.complexorder = list()
        # Complex: [...["Lips", "LipsAperture"], ["Lips", "LipsProtrusion"]...]
        for n in self.superficialorder:
            if not self.complexarticulators.has_key(n):
                self.lowerorder.append(n)
                self.complexorder.append(n)
            else:
                for part in self.complexarticulators[n]:
                    self.lowerorder.append(part[0])
                    self.complexorder.append([n, part[0]])

    
    def fetch_missing_articulators(self, articulators):
        """For one articulator or a set of articulators, returns the
            complementary set to cover the full vocal tract, handling the
            short/full articulator's names and complex articulators.
        Input:
            articulators: string or list of string: name of an articulator
                or of a part of it or a list of articulators' names   
        Output:
            missingarts: list of string: the complementary list of names of
                articulators or their parts (e.g. "Tongue", "LipsProtrusion").
        """
        if articulators == list():
            return self.superficialorder
        if articulators != list(articulators): # It is actually one articulator name
            articulators = [articulators]
        # First, we need to determine if the names provided in articulators are
        # full or short.
        isfull = True if self.fullnames.has_key(articulators[0]) else False
        # We convert all names to full ones.
        for n, art in enumerate(articulators):
            try:
                articulators[n] = self.shortnames[art]
            except:
                if self.fullnames.has_key(art):
                    pass
                else:
                    print "Articulator \'"+art+"\' has not been recognized."
                    return None
        missingarts = list(self.superficialorder)
        articulatorparts = list()
        for art in articulators:
            if art in missingarts: # i.e. it is not a subpart of a complex articulator
                missingarts.remove(art)
            else: # art is a subpart of a complex articulator.
                articulatorparts.append(art)
        # All simple articulators have been processed.
        if articulatorparts:
            # Complex articulators need further handling.
            for complart, parts in self.complexarticulators.iteritems():
                matching = [part[0] for part in parts if part[0] in articulatorparts]
                if matching:
                    complement = [part[0] for part in parts if part[0] not in matching]
                    if complart in missingarts:
                        missingarts.remove(complart)
                        missingarts.extend(complement)
            # Now, reorder the result.
            missingarts = self.reorder_articulators(missingarts)
        if isfull:
            return missingarts
        for n, missing in enumerate(missingarts):
            missingarts[n] = self.fullnames[missing]
        return missingarts
    
    
    def reorder_articulators(self, articulators, vocal=True):
        """Returns a correctly reordered list of articulators.
        Input: list of strings (names of articulators or their subparts,
            either full or short)
            vocal: whether to print warnings about encountered ambiguous
                orderings. By default, True (yes, to print).
        Output: reordered input list.
        """
        res = list()
        for el in self.complexorder:
            if el != list(el):
                if el in articulators:
                    res.append(el)
            else:
                if el[0] in articulators:
                    if el[0] not in res:
                        res.append(el[0])
                    if el[1] in articulators:
                        if vocal:
                            print "There are various orderings possible in " + \
                                str(articulators) + "!"
                        res.append(el[1])
                elif el[1] in articulators:
                    res.append(el[1])
                # The depth is assumed to be 2.
        return res
    
    
    def parameter_numbers(self, articulators):
        """For a given set of articulators' names, returns the set of
            articulatory parameters they are responsible for.
        Input:
            articulators: a string or list of strings: names of articulators,
                in their full or short forms.
        Output:
            list of integers: which parameters they are responsible for.
        """
        if articulators != list(articulators):
            return self.articulators[articulators]
        return sorted(set([par for name in articulators for par in self.articulators[name]]))


class ArticulatoryVector(object):
    def __init__(self, artvec, ph=None, vocaltract=VocalTract()):
        self.vt = vocaltract
        self.vector = list(artvec)
        self.phoneme = ph           # See griddecipher in Utterance.temporal_grid
    
    def __getitem__(self, key):
        if not isinstance(key, basestring):
            return self.vector[key]
        elif self.vt.articulators.has_key(key):
            return [self.vector[parnum] for parnum in self.vt.articulators[key]]
        else:
            return None
    
    def __setitem__(self, key, val):
        if not isinstance(key, basestring):
            updatedvector = self.vector
            updatedvector[key] = val
            setattr(self, 'vector', updatedvector)
        elif self.vt.articulators.has_key(key) and len(val)==len(self.vt.articulators[key]):
            for parnum in self.vt.articulators[key]:
                self.vector[parnum] = val[parnum - self.vt.articulators[key][0]]
        else:
            print "ArticulatoryVector.__setitem__({}, {}): an error occurred.".format(key, val)
    
    def __str__(self):
        return ";\n".join(["{}: ".format(art) + ", ".join(["{0:.3f}".format(self.vector[i]) for i in self.vt.articulators[art]]) for art in self.vt.superficialorder])
    
    def to_list(self):
        return self.vector
    
    def __eq__(self, other):
        return all(np.isclose(self.vector, other.vector))
    
    def __len__(self):
        return len(self.vector)
    
    
class Phoneme(object):
    """A class for relating the phoneme database files to the phonemes in
        speech synthesis.
    Language: French
    Model: "Data/VT-Model/antoineIRM.pca"
    Captures: "Data/VT-Model/Contours-to-Build-The-Model/*.[ctr|pgm]"
    Visualisation: "Data/Speech-Synthesis-Database/Images/*.png"
    """
    def __init__(self, name,
                 folderpath = "Data/Speech-Synthesis-Database/DB-Full/",
                 extension = ".artv"):
        """Parses the file with a given phoneme.
        Input:
            name:string, the name of the phoneme;
            folderpath:string, the location of the database with the phonemes.
                    By default, folderpath is set to
                    "Data/Speech-Synthesis-Database/DB-Full/".
            extension:string, the extension of the phoneme files.
                    By default, extension is set to ".artv" which is the format
                    of the database after its expansion.
        Parsing the file and creating an object description:
            When parsing the file, it is assumed that comments are marked by
                the "//" sign, and all information after it is discarded.
            Aside from the attributes which are shared between all the members
            of the same implementation version, we mark:
            - name: as stored in the first line of the phoneme file. Spacing is
                removed.
                The result is stored in
                *   self.name.
            - articulatory features separated by the " + " - in the second
                line. The system may use not all of them. Examples of such
                features are: Vowel, Consonant, Semivowel (which is actually
                treated as Consonant); Fricatives, Affricates, Stops...;
                Open, Closed...; Bilabial, Dental, Alveolar...
                The result is stored in the following attributes:
                *   self.phclass ("V" for vowel, "C" for consonant, or
                        "SILENCE")
                *   self.voicing (True for voiced phonemes, False for
                        voiceless)
                *   self.artmanner (only in case of consonants: articulation
                        manner)
                *   self.artfeatures: any other features, such as place of
                        articulation.
                *   self.placeofarticulation: int - the rank in place of
                        culation: the greater it is, the more front movement
                        the phoneme involves.
            - critical articulators or parts of articulators, separated by the
                " + " - in the third line. They are stored as a list in the
                following attribute:
                *   self.critart: list of strings
            - Average duration of the phoneme when not long:
                *   self.duration
            - A comment line for the sake of traceability in the database (l.5)
            - Articulatory vectors: since the system treats vowels and
                consonants differently, the contents of this dictionary will
                depend on the nature of the phoneme.
                If it is a vowel or the special case SILENCE (the natural,
                silent position of the vocal tract), then the dictionary
                attribute
                *   self.artv
                will have only one key: "Solo". This is the captured vocal
                tract configuration for the phoneme without any context:
                self.artv["Solo"]:list of articulatory parameter values (float)
                If this is a consonant, the file from line 5 to the penultimate
                one is filled with vectors of this consonant anticipating
                particular vowels. The usual line will look like this:
                "z + i : 0.18 (...) -0.22 // from the voiceless s + i, S62":
                In such an example, "i" becomes a key in the self.artv
                dictionary, and the value is the vector [0.18, ..., -0.22]:
                self.artv[vowel:str] : list of articulatory parameter values
                The comment following each vector is not parsed: its sole
                purpose is, again, traceability of the database.
            - Projections: it is argued that consonant positions when
                anticipating particular vowels can be estimated from the
                positions when anticipating corner vowels: [u], [i], and [a].
                This process requires a correspondence between these particular
                vowels and the corner ones, which is exactly what is stored in
                the 5th line of the document if the document describes a vowel.
                A usual line may look like the following example:
                "0.518248, 0.200141 //
                        Projections: y represented as u + s (i - u) + q (a - u)"
                So, if the phoneme is a vowel, it has the following attribute:
                *   self.proj = [s:float, q:float]
                
        A few side remarks:
        - The parameters in the articulatory vectors are to be separated from
        each other by a *single* space character.
        - The s and q coefficients for relating vowels to [u], [i], and [a] are
        to be separated from each other by ", ".
        - The program does not expect to find an empty line in the input file.
        """
        phonemepath = folderpath + name + extension
        if not os.path.exists(phonemepath):
            print "There is no phoneme [{}] in {}.".format(name, folderpath)
            self.status = "ERROR"
            return
        self.status = "+"
        with open(phonemepath, 'r') as phfile:
            contents = phfile.readlines()
            cleanupafterwindows = list()
            for c in contents:
                cleanupafterwindows.append(c.replace("\r", ""))
            contents = cleanupafterwindows
            # Line 1 (contents[0]): the name of the phoneme
            self.name = contents[0][:contents[0].find("//")].replace(" ", "")
            # Line 2 (contents[1]): e.g. Vowel + Front + Open + Rounded
            # This is the place for phonetic description
            features = contents[1][:contents[1].find("//")].replace(" ", "").split("+")
            if "Vowel" in features: # <------ Vowels / Consonants / Silence
                self.phclass = "V"
                self.voicing = True # <------ Voicing
            elif "Consonant" in features:
                self.phclass = "C"
                if "Voiced" in features:
                    self.voicing = True
                else:
                    self.voicing = False
            else:
                self.phclass = "SILENCE"
                self.artmanner = "SILENCE"
                self.voicing = False
            if self.phclass == "C": # <------ Manner of articulation, consonants
                manner = list_intersect(["Stop", "Affricate", "Fricative",
                                            "Nasal", "Flap", "Liquid", "Semivowel",
                                            "Approximant", "Trill"], features)
                if manner:
                    manner = manner[0]
                self.artmanner = manner
            # The current implementation disregards any more fine-grained
            # information.
            discard = ["Vowel", "Consonant", "Voiced", "Voiceless",
                       "Stop", "Affricate", "Fricative", "Nasal", "Flap",
                       "Approximant", "Liquid", "Trill",
                       "Sibilant", "Semivowel", "Lateral"]
            for el in discard:
                if el in features:
                    features.remove(el)
            self.artfeatures = features # <------ Usually: place of articulation
            # Places of articulation are built for the French language!
            # For other languages, they will have to be adapted.
            # The greater self.placeofarticulation is, the more frontal
            # movement this phoneme requires.
            if self.phclass == "V":
                if "Rounded" in self.artfeatures:
                    if "Close" in self.artfeatures:
                        self.placeofarticulation = 8
                    else:
                        self.placeofarticulation = 7
                elif "Front" in self.artfeatures:
                    self.placeofarticulation = 4
                elif "Nazalized" in self.artfeatures:
                    self.placeofarticulation = 1
                else:
                    self.placeofarticulation = 3
            elif self.phclass == "C":
                if "Bilabial" in self.artfeatures:
                    self.placeofarticulation = 8
                elif "Labialized" in self.artfeatures or \
                        "Labio-dental" in self.artfeatures or \
                            "Labio-velar" in self.artfeatures:
                    self.placeofarticulation = 7
                elif "Dental" in self.artfeatures:
                    self.placeofarticulation = 6
                elif "Palato-alveolar" in self.artfeatures:
                    self.placeofarticulation = 5
                elif "Palatal" in self.artfeatures:
                    self.placeofarticulation = 4
                elif "Velar" in self.artfeatures:
                    self.placeofarticulation = 2
                elif "Uvular" in self.artfeatures:
                    self.placeofarticulation = 1
            # Line 3 (contents[2]): e.g. Velum + Tongue - critical articulators
            # It is also possible to note them in their short form: Vel + T
            # See the VocalTract class for further information
            self.critart = contents[2][:contents[2].find(
                                        "//")].replace(" ", "").split("+")
            etalonearts = VocalTract().articulators # Both full and short names
            for art in self.critart:
                if not etalonearts.has_key(art):
                    print "The {} articulator in phoneme {} is not recognised.".format(art, self.name)
            # Line 4 (contents[3]): how many milliseconds the phoneme
            # lasts on average, e.g."40 ms". This line may be left empty.
            duration = re.findall("^\d+", contents[3])
            if duration:
                self.duration = int(duration[0])
            else:
                if self.phclass == "V":
                    self.duration = 90    # Average vowel duration is 90 ms
                elif self.phclass == "C":
                    self.duration = 50    # Average consonant duration is 50 ms
            # Line 5 (contents[4]): a comment line.
            # Lines 6-... (contents[5:]): depends on the phoneme class.
            # Vowels:
            #   * Line 6: vowelname : 1.00 -2.14 ... 3.14 // S...,
            #         i.e. articulatory parameters are separated by spaces,
            #         and there may be a comment after "//"
            #         This is the configuration of this vowel on its own.
            #   * Line 7: 0.15, 0.65 // Projections,
            #         i.e. two floats separated by comma and a space, possibly
            #         followed by a comment after "//"
            #         These are the projections of this vowel onto the corner
            #         vowels. They are used to estimate the missing samples in
            #         the database.
            #   EOF.
            # Consonants:
            #   * Any line: consonantname + context : 1.00 2.03 ... // S...,
            #         i.e. the target configuration for the vocal tract to
            #         pronounce this consonant in a given context.
            #   There also is an estimation for the consonant on its own,
            #   currently from the contexts when preceding an [a] and an [i].
            # Semivowels: treated as consonants, but can be anticipated as
            #         vowels.
            #   * Line 6: name + context : -0.24 0.85 ... // S...,
            #         i.e. the target configuration for the vocal tract to
            #         produce this semivowel in a given context.
            #         A copy of this line is also used to build an estimation
            #         of this semivowel on its own.
            #   * Line 7: 0.13, 0.04 // Projections: ...
            #         i.e. two floats separated by comma and a space, possibly
            #         followed by a comment after "//"
            #         These are the projections of this vowel onto the corner
            #         vowels. They are used to estimate the missing samples in
            #         the database.
            # Silence:
            #   * Line 6 in Vowels, and then EOF.
            #         Silence is used for the articulatory configuration at the
            #         beginning and the end of an utterance.
            self.artv = dict()
            if self.phclass in ["V", "SILENCE"]:
                self.artv = dict()
                strli = contents[5][:contents[5].find(
                                        " //")].split(" : ")[1].split(" ")
                self.artv["Solo"] = ArticulatoryVector([float(el) for el in strli])
            else:
                if self.artmanner != "Semivowel":
                    limit = len(contents)
                else:
                    limit = 6
                for line in contents[5:limit]:
                    configuration = line[:line.find(" //")].split(" : ")
                    anticipating = configuration[0].split(" + ")[1]
                    strli = configuration[1].split(" ")
                    self.artv[anticipating] = ArticulatoryVector([float(el) for el in strli])
                if self.artv.has_key("a") and self.artv.has_key("i"):
                    self.artv["Solo"] = ArticulatoryVector([float("{0:.2f}".format(0.8*aval + \
                                        0.2*ival)) for aval, ival in zip( \
                                        self.artv["a"], self.artv["i"])])
                    # It is presumed that the consonant by itself can be
                    # estimated from its anticipatory positions for [a] and [i]
                elif self.artmanner == "Semivowel":
                    # Semivowels were recorded only in one context.
                    if self.artv.has_key("a"):
                        self.artv["Solo"] = self.artv["a"]
                    elif self.artv.has_key("i"):
                        self.artv["Solo"] = self.artv["i"]
            if self.phclass == "V" or self.artmanner == "Semivowel":
                strli = contents[6][:contents[6].find(" //")].split(", ")
                self.proj = [float(el) for el in strli]
    
    
    def slice(self, articulators, key="Solo"):
        """Returns a list of the parameters for a particular articulator in an
        articulatory vector or a list of articulatory vectors.
        Input:
            articulators: string or list of strings. Expected arguments: see the
                description of the constructor in the VocalTract class (namely,
                the articulators attribute).
            key: string. It can be the name of the phoneme that is anticipated,
                or "Solo" to get the pure vowel or consonant configuration.
                By default, it is set to "Solo".
        Output: vector of float.
        """
        if articulators != list(articulators):
            articulators = [articulators]
        vt = VocalTract()
        arts = vt.articulators
        articulators = vt.reorder_articulators(articulators)
        res = list()
        for sliceart in articulators:
            if arts.has_key(sliceart):
                if self.artv.has_key(key):
                    res.extend([self.artv[key][k] for k in arts[sliceart]])
            else:
                print "Unknown articulator \""+sliceart+"\" in ["+self.name+"]."
        return res


class Syllable(object):
    """A class for taking the necessary phonemes for a syllable. The
        articulatory targets are not produced at this stage, though,
        because findings show that coarticulatory effects can span over
        syllable boundaries.
    Language: French
    Model: "Data/VT-Model/antoineIRM.pca"
    Captures: "Data/VT-Model/Contours-to-Build-The-Model/*.[ctr|pgm]"
    Visualisation: "Data/Speech-Synthesis-Database/Images/*.png"
    """
    def __init__(self, sylltext,
                 folderpath="Data/Speech-Synthesis-Database/DB-Full/",
                 extension=".artv"):
        """
        Input:
            sylltext:string, the syllable to be processed; multicharacter
                phonemes are to be put in curly brackets: "{...}". For other
                special characters, see below.
            folderpath:string, the location of the database with the phonemes.
                By default, folderpath is set to
                    "Data/Speech-Synthesis-Database/DB-Full/".
            extension:string, the extension of the phoneme files.
                    By default, extension is set to ".artv" which is the format
                    of the database after its expansion.
        Obtains a sequence of articulatory targets for the given input syllable
            and puts them into the "targets" attribute:
            *   self.constinuents = list of Phoneme
            *   self.anticipated = list of keys for the "artv" attribute of
                    Phonemes in self.constituents. An example for a CCVC-syllable:
                    self.targets = ["a", "a", "Solo", "Solo"]
                    This means that the first articulatory vector that can be
                    used as the target is /TheFirstConsonant/.artv["a"],
                    and then come /TheSecondConsonant/.artv["a"],
                    /Vowel/.artv["Solo"] (i.e. the pure vowel position), and,
                    finally, /Consonant/.artv["Solo"].
                    These target vectors will be fed to the coarticulation
                    model.
        There are special markers allowed in the sequence. The parsing
        algorithm assumes the following order:
        1)  Stress-related symbols (if excluded, the syllable is not stressed)
        2)  Melody-related symbols (if excluded, the intonation contour is
                even)
        3)  Phonemes in the syllable
        4)  Phoneme that is intended long should be immediately followed by the
                prolongation-related symbol (if there is none, the phoneme is
                of regular duration)
        In the literature, there are a lot of types of stress that are used:
        rhythmic, syntagmatic, secondary, emphatic...; there are various
        degrees of phoneme prolongation; there are a lot of melodic patterns to
        mark.
        However, this implementation makes use of the bare minimum:
        -   '   The normal rhythmic stress which falls onto the last syllable
                    of a rhythmic group.
        -   /   Rising intonation
        -   _   Even intonation contour
        -   \   Falling intonation (since it is a special character, it has to
                    be escaped: \\)
        -   :   Long phoneme (vowel or consonant; can be used for gemminated
                    stops too).
        The corresponding attributes store this information:
            *   self.contour: string. Contains graphics for intonation, e.g.
                    \_/, _/, \. If no information has been given,
                    self.contour = "_" (even).
            *   self.stress: boolean. Whether the current syllable is stressed.
            *   self.lengths: list of strings. The elements are in one-to-one
                    relation with the phonemes in the syllable. "Reg" means
                    regular phoneme duration, "Long" means long.
        If you mark the boundary between syllables in the user input, then
            the syllable may begin by a sequence for the intonation contour,
            then there may be an accent sign, and then should come the phoneme
            names, e.g.: "/_'{zh}u:r".
        If you rely on the syllable segmentation algorithm, the stress mark has
            to be immediately preceding the vowel it is on.
        """
        self.constituents = list()
        if (len(re.findall("([/_\\\\]+)", sylltext)) > \
                len(re.findall("^([/_\\\\]+)", sylltext))) \
                    or (len(re.findall("'{1}", sylltext)) > \
                        len(re.findall("^([/_\\\\]*)('{1})", sylltext))):
            self.status = "TO SPLIT"
            sylltext = str(sylltext).replace("/", "")
            sylltext = sylltext.replace("\\", "")
            sylltext = sylltext.replace("_", "")
            sylltext = sylltext.replace(":", "")
            sylltext = sylltext.replace("'", "")
            sylltext = sylltext.replace("{oport}", "{o_port}") # CAUTION
        contour, stress = re.findall("^([/_\\\\]*)('{0,1})", sylltext)[0]
        # Any combination of /, \, and _, positioned at the beginning of
        # sylltext.
        # \ can be either escaped or not, but if it is not escaped, it can
        # unexpectedly get merged with the following symbol. "\\" counts as one
        # symbol.
        # To be used in the mode when the partition into syllables is available
        if contour:
            self.contour = contour
            sylltext = sylltext.replace(contour, "")
        else:
            self.contour = "_"
        if stress:
            self.stress = True
            sylltext = sylltext.replace(stress, "")
        else:
            self.stress = False
        self.lengths = list()
        for m in re.split("({.*?})", sylltext):
            if not (m.startswith("{") and m.endswith("}")):
                for ch in list(m):
                    if ch == ":":
                        self.lengths[-1] = "Long"
                        self.constituents[-1].duration *= 2.7
                        self.constituents[-1].duration = \
                            int(self.constituents[-1].duration)
                        continue
                    self.constituents.append(Phoneme(ch,folderpath,extension))
                    self.lengths.append("Reg")
            else:
                self.constituents.append(Phoneme(m[1:-1],folderpath,extension))
                self.lengths.append("Reg")
        for ph in self.constituents:
            if ph.status == "ERROR":
                print "There is an unrecognised phoneme in your input."
                print "\'{\' and \'}\' should enclose all multi-character phonemes."
                self.status = "ERROR"
                return
        try:
            if self.status == "TO SPLIT": # self.status may be not defined yet
                return
        except:
            self.status = "+"

    
    def express(self, vocal=True, anticipations=list()):
        """Prints the constructed syllable in terms of its constituents
        Input:  vocal: boolean - whether to print the output. By default, yes
                        (True).
                anticipations: list of keys for the artv attributes in Phoneme.
                        When empty, the usual transcription will be returned.
                        By default, it is empty.
        Output: string, the textual representation of the syllable.
        """
        res = str()
        if self.status == "ERROR":
            return res
        if self.contour != "_":
            res += self.contour
        if self.stress:
            res += "'"
        for n, ph in enumerate(self.constituents):
            if len(ph.name) > 1:
                res += "{"+ph.name+"}"
            else:
                res += ph.name
            if anticipations:
                if anticipations[n] != "Solo":
                    res += "("+anticipations[n]+") "
                else:
                    res += " "
            if self.lengths[n] == "Long":
                if anticipations:
                    res = res[:-1] + ": "
                else:
                    res += ":"
        if vocal:
            print res
        elif anticipations:
            return res[:-1]
        return res


    def is_polyvocalic(self):
        """Determines if the syllable is actually a syllable
        or a merging of phonemes.
        Output: boolean: True for "it is actually several syllables",
                        False for "it is one syllable"
        """
        vowelcount = len([ph for ph in self.constituents \
                              if ph.phclass == "V"])
        if vowelcount > 1:
            return True
        else:
            return False
    
    
    def split(self, syntext,
              folderpath="Data/Speech-Synthesis-Database/DB-Full/",
              extension=".artv"):
        """Splits a syllable that was considered one erroneously into more
        syllables.
        Input: syntext: text with markings on the intonation and stress, but
            without separation into syllables.
            folderpath = "Data/Speech-Synthesis-Database/DB-Full/",
            extension = ".artv"
            An example:
            * The correct syntexts (with the intonation contour, long phonemes,
                stress and syllables):
                "\\l{epsilon}-\\za-\\ba-/_'{zh}u:r", "s{o~}-t{o~}m-\\'be"
            * Without syllables it would be:
                "\\l{epsilon}\\za\\ba/_{zh}'u:r", "s{o~}t{o~}m\\b'e"
        Output: a list of Syllable.
        """
        # The rules (Automatic detection of syllable boundaries in spontaneous
        # speech - B. Bigi et al.):
        # V for vowels, C for consonants,
        # G for glides, L for liquids, F for fricatives, S for stops:
        #       VV      =>      V-V
        #       VCV     =>      V-CV
        #       VCCV    =>      VC-CV
        #                if not V-CGV
        #                   or  V-FLV
        #                   or  V-SLV
        #       VCCCV   =>      VC-CCV
        #                if not V-FLGV
        #                   or  V-SLGV
        #                   or  VSL-SV
        #       VCCCCV  =>      VC-CCCV
        #       VCCCCCV =>      VCC-CCCV
        if self.status == "ERROR":
            return None
        newsyllables = list()
        currnewsyllable = list()
        currmarkers = list()
        for ph in self.constituents:
            currnewsyllable.append(ph)
            if ph.phclass == "V":
                currmarkers.append("V")
                prevvowelpos = currmarkers.index("V")
                currvowelpos = len(currmarkers)-1
                if currvowelpos == prevvowelpos: # The first vowel
                    continue
                if currvowelpos-prevvowelpos == 1:              # V-V
                    breakpos = currvowelpos
                elif currvowelpos-prevvowelpos == 2:            # V-CV
                    breakpos = currvowelpos-1
                elif currvowelpos-prevvowelpos == 3:
                    if currmarkers[currvowelpos-1] == "G":      # V-CGV
                        breakpos = currvowelpos-2
                    elif currmarkers[currvowelpos-1] == "L" \
                        and currmarkers[currvowelpos-2] == "F": # V-FLV
                        breakpos = currvowelpos-2
                    elif currmarkers[currvowelpos-1] == "L" \
                        and currmarkers[currvowelpos-2] == "S": # V-SLV
                        breakpos = currvowelpos-2
                    else:
                        breakpos = currvowelpos-1               # VC-CV
                elif currvowelpos-prevvowelpos == 4:
                    if currmarkers[currvowelpos-1] == "G" \
                        and currmarkers[currvowelpos-2] == "L" \
                            and currmarkers[currvowelpos-3] == "F":
                        breakpos = currvowelpos-3               # V-FLGV
                    elif currmarkers[currvowelpos-1] == "G" \
                        and currmarkers[currvowelpos-2] == "L" \
                            and currmarkers[currvowelpos-3] == "S":
                        breakpos = currvowelpos-3               # V-SLGV
                    elif currmarkers[currvowelpos-1] == "S" \
                        and currmarkers[currvowelpos-2] == "L" \
                            and currmarkers[currvowelpos-3] == "S":
                        breakpos = currvowelpos-1               # VSL-SV
                    else:
                        breakpos = currvowelpos-2               # VC-CCV
                elif currvowelpos-prevvowelpos == 5:
                    breakpos = currvowelpos-3                   # VC-CCCV
                else:
                    breakpos = currvowelpos-3                   # VCC-CCCV
                newsyllables.append(currnewsyllable[:breakpos])
                currnewsyllable = currnewsyllable[breakpos:]
                currmarkers = currmarkers[breakpos:]
            else:
                if ph.artmanner == "Semivowel":
                    currmarkers.append("G")
                elif ph.artmanner == "Liquid":
                    currmarkers.append("L")
                elif ph.artmanner == "Fricative":
                    currmarkers.append("F")
                elif ph.artmanner == "Stop":
                    currmarkers.append("S")
                else:
                    currmarkers.append("C")
        if currnewsyllable:
            newsyllables.append(currnewsyllable)
        locators = list()
        for syll in newsyllables:
            locators.append(list())
            for ph in syll:
                phname = "{"+ph.name+"}" if len(ph.name) > 1 else ph.name
                locators[-1].append(phname)
        matches = re.findall("([/_\\\\]*)('{0,1})({[A-Za-z~]*}|[A-Za-z]*)(:{0,1})",
                             syntext)
        sylltext = str()
        conttext = str()
        accenttext = str()
        currsyll = 0
        res = list()
        for c, a, phs, dur in matches:
            if c == "" and a == "" and phs == "" and dur == "":
                continue
            sylltext = sylltext + phs + dur
            conttext = conttext + c
            accenttext = accenttext + a
            if phs.startswith("{") and phs.endswith("}"):
                if locators[currsyll].index(phs) == 0:
                    locators[currsyll].remove(phs)
                else:
                    print "Error in syllable segmentation."
                    return None
            else:
                seq = list(phs)
                for q, ph in enumerate(seq):
                    if locators[currsyll]:
                        if locators[currsyll].index(ph) == 0:
                            locators[currsyll].remove(ph)
                        else:
                            print "Error in syllable segmentation."
                            return None
                    else:
                        # phs contains a collection of single-character-named
                        # phonemes from several syllables.
                        # We have to end the current syllable here and start
                        # anew.
                        currsyll += 1
                        if locators[currsyll].index(ph) == 0:
                            locators[currsyll].remove(ph)
                        else:
                            print "Error in syllable segmentation."
                            return None
                        sylltextold = conttext + accenttext + \
                                        sylltext[:sylltext.find(phs)] + \
                                        "".join(phs[:q])
                        #res.append(Syllable(sylltextold, folderpath, extension))
                        newsyll = Syllable(sylltextold, folderpath, extension)
                        res.append(newsyll)
                        sylltext = "".join(phs[q:]) + dur
                        conttext = str()
                        accenttext = str()
            if locators[currsyll] == list():
                currsyll += 1
                sylltext = conttext + accenttext + sylltext
                res.append(Syllable(sylltext, folderpath, extension))
                sylltext = str()
                conttext = str()
                accenttext = str()
        return res

        
class Syntagm(object):
    """A class for processing whole rhythmic groups in an utterance and
        constructing articulatory targets according to the assumptions of the
        implemented coarticulation model and taking note of coarticulatory
        effects that span over syllable boundaries.
    Language: French
    Model: "Data/VT-Model/antoineIRM.pca"
    Captures: "Data/VT-Model/Contours-to-Build-The-Model/*.[ctr|pgm]"
    Visualisation: "Data/Speech-Synthesis-Database/Images/*.png"        
    """
    def __init__(self, syntext,
                 folderpath="Data/Speech-Synthesis-Database/DB-Full/",
                 extension=".artv"):
        """
        Input:
            syntext:string, the rhythmic group being processed.
                Boundaries between syllables are to be marked by hyphen, "-".
                You can also rely on the syllable segmentation algorithm
                implemented within the Syllable class
                (see Syllable().split()) and not mark any boundaries between
                syllables at all, or mark it only at a place where the
                algorithm makes a mistake.
                There are syntax rules for syllables: see documentation of the
                    Syllable class, applying to the case of syllable
                    segmentation being marked by user or by the program.
            folderpath:string, the location of the database with the phonemes.
                By default, folderpath is set to
                    "Data/Speech-Synthesis-Database/DB-Full/".
            extension:string, the extension of the phoneme files.
                    By default, extension is set to ".artv" which is the format
                    of the database after its expansion.
        """        
        self.sylls = [Syllable(s, folderpath,
                               extension) for s in syntext.split("-")]
        for syll in self.sylls:
            if syll.status == "TO SPLIT":
                break
            if syll.status == "ERROR":
                self.status = "ERROR"
                print "\'-\' should separate syllables."
                return
        polyvocalic = list()
        syllablecorrections = list()
        for syllable, sylltext in zip(self.sylls, syntext.split("-")):
            if syllable.is_polyvocalic() or syllable.status == "TO SPLIT":
                polyvocalic.append(syllable)
                syllablecorrections.append(syllable.split(sylltext))
                #   Either the user has put no syllable segmentation signs,
                #   relying on the further algorithm of syllable segmentation,
                #   or there has been a mistake (since we treat semivowels as
                #   consonants and diphthongs are two vowels in two different
                #   syllables, it should not have occurred).
        for error, correction in zip(polyvocalic, syllablecorrections):
            k = self.sylls.index(error)
            try:
                self.sylls = self.sylls[:k] + correction + self.sylls[k+1:]
            except: # k+1 inaccessible
                self.sylls = self.sylls[:k] + correction
        for syll in self.sylls:
            if syll.status == "ERROR":
                self.status = "ERROR"
                print "\'-\' should separate syllables."
                return
        self.status = "+"
        # It is necessary to build articulatory targets.
        # The first part of this is to decide which entries from the database
        # to use.
        # In general, we assume that all vowels have pure configuration
        # as their target (this simplification comes from the fact that
        # contexts for vowels are not available), and all consonants anticipate
        # the most imminent vowel. But there are several points to mark:
        #       *   Syllable boundaries: they do not prevent anticipation of
        #               the vowel, but coarticulatory effects are less
        #               prominent across them.
        #       *   Time window: at what longest advance it is possible to
        #               anticipate a coming vowel.
        #       *   Stack limits: over how many phonemes coarticulatory
        #               effects can propagate.
        #       *   Places of articulation are organized from back to the
        #               front. If a phoneme sequence requires moving to the
        #               front, then to the back, and then to the front again,
        #               this back constituent will hinder articulators'
        #               anticipation of the later front constituent before the
        #               back one comes.
        #       *   The articulatory organs that are critical for production of
        #               an imminent sound start anticipation earlier in order
        #               to have more time to reach the target.
        self.anticipations = list() # List of keys for the artv attributes
        currvowel = None # The vowel or semivowel which is the current
                         # candidate for anticipation
        whichsyllable = 0 # Marker for origin of currvowel
        currtime = 0 # How long from the current consonant until currvowel
        currcount = 0 # How many consonants from the current consonant until
                      # currvowel
        block = False # It is not possible to start anticipating a phoneme
                      # and then reconsider. So, if there is a phoneme which
                      # prevents anticipation of currvowel, it blocks
                      # its anticipation for all phonemes before it.
        neighbourplace = 0 # For the place of articulation of the following
                      # consonant
        for n, syll in enumerate(reversed(self.sylls)):
            for ph in reversed(syll.constituents):
                if currvowel == None:
                    self.anticipations.append("Solo")
                    if ph.phclass == "V" or ph.artmanner == "Semivowel":
                        currvowel = ph
                        whichsyllable = n # Identifier for the vowel origin
                else: # We have already encountered a vowel.
                      # All previous (and in our order, subsequent) consonants
                      # that pass the constraints should anticipate it.
                    if ph.phclass == "V" or ph.artmanner == "Semivowel":
                        currvowel = ph
                        whichsyllable = n
                        currtime = 0
                        currcount = 0
                        block = False
                        neighbourplace = 0
                        self.anticipations.append("Solo")
                    elif ph.phclass == "C":
                        currtime += ph.duration
                        currcount += 1
                        if block or currtime >= 450 or currcount > 5:
                            # block is True if there is a consonant (later)
                            # preventing anticipation of currvowel.
                            # The limit for anticipation is set 200 ms.
                            # The limit size of the anticipation stack is set 5
                            self.anticipations.append("Solo")
                            neighbourplace = ph.placeofarticulation
                            continue
                        # Not all vowel anticipations are allowed to cross the
                        # boundary between syllables.
                        if whichsyllable != n:
                            """print "Vowel {}, ph {} => {}".format(
                                currvowel.placeofarticulation,
                                ph.placeofarticulation,
                                currvowel.placeofarticulation - ph.placeofarticulation)
                            """
                            if currvowel.placeofarticulation - \
                                ph.placeofarticulation > 0 and not block:
                                self.anticipations.append(currvowel.name)
                            else:
                                block = True
                                self.anticipations.append("Solo")
                                continue
                        else:
                            if neighbourplace:
                                currpl = ph.placeofarticulation
                                gapcons = neighbourplace - currpl
                                gapvow = currvowel.placeofarticulation - currpl
                                if gapvow*gapcons > 0 or abs(gapvow-gapcons) <= 5:
                                    self.anticipations.append(currvowel.name)
                                else:
                                    self.anticipations.append("Solo")
                                    block = True
                            else:
                                self.anticipations.append(currvowel.name)
                            neighbourplace = ph.placeofarticulation
        self.anticipations.reverse()                
        temp = list(self.anticipations)
        self.anticipations = list()
        k = 0
        for syll in self.sylls:
            self.anticipations.append(list())
            for ph in syll.constituents:
                self.anticipations[-1].append(temp[k])
                k += 1

 
    def express(self, vocal=True, showcoart=False):
        """Prints the constructed syntagm in terms of syllables it consists of.
        Input:  vocal: boolean - whether to print the output. By default, yes
                        (True).
                showcoart: whether to include which are the most prominently
                        anticipated vowels and semivowels in the transcription.
                        By default, not (False).
        Output: string, the textual representation of the syntagm.
        """
        if self.status == "ERROR":
            return str()
        if not showcoart:
            syllableexpressions = [syll.express(False) for syll in self.sylls]
            res = "-".join(syllableexpressions)
        else:
            syllableexpressions = [syll.express(False, self.anticipations[k]) \
                                   for k, syll in enumerate(self.sylls)]
            res = " - ".join(syllableexpressions)
        if vocal:
            print res
        else:
            return res


class Utterance(object):
    """A class for breaking the utterance into rhythmic groups and syllables,
        forming articulatory targets within the assumptions of the
        implemented coarticulation model, and writing all the files that are
        necessary to synthesize the given utterance.
    Language: French
    Model: "Data/VT-Model/antoineIRM.pca"
    Captures: "Data/VT-Model/Contours-to-Build-The-Model/*.[ctr|pgm]"
    Visualisation: "Data/Speech-Synthesis-Database/Images/*.png"
    """
    def __init__(self, uttertext, folderpath="Data/Speech-Synthesis-Database/DB-Full/",
                 extension=".artv"):
        """
        Input:
            uttertext:string, the utterance being processed. Boundaries between
                rhythmic groups / syntagms (the current implementation does not
                differentiate between various pauses) are to be marked by
                " | ", i.e. vertical line | enclosed in spaces on both sides.
                There are syntax rules for syntagms and syllables: see
                    documentation of the Syntagm and Syllable
                    classes respectively.
            folderpath:string, the location of the database with the phonemes.
                By default, folderpath is set to
                    "Data/Speech-Synthesis-Database/DB-Full/".
            extension:string, the extension of the phoneme files.
                    By default, extension is set to ".artv" which is the format
                    of the database after its expansion.
        """
        self.synts = [Syntagm(s, folderpath,
                            extension) for s in uttertext.split(" | ")]
        # At this stage, it is possible to add modifications to the pre-defined
        # phoneme durations, taking into account all factors.
        for synt in self.synts:
            if synt.status == "ERROR":
                print "\' | \' should separate rhythmic groups."
                self.status = "ERROR"
                return
        self.status = "+"


    def express(self, vocal=True, showcoart=False, onlytextual=False,
                limit=None, protectcurlybraces=False):
        """Prints a constructed utterance in terms of syntagms it comprises.
        Input:  vocal: boolean - whether to print the output. By default, yes
                        (True).
                showcoart: whether to include which are the most prominently
                        anticipated vowels and semivowels in the transcription.
                        By default, not (False).
                onlytextual: whether to strip the resulting string of all
                        special characters. By default, not (False).
                limit: integer: what is the maximally allowed length of the
                        result. If None, no limit is forced. By default,
                        None.
                protectcurlybraces: boolean: whether to protect the "{}" signs
                        with the purpose of further usage of string.format().
                        By default, no.
        Output: string, the textual representation of the utterance.
        """
        if self.status == "ERROR":
            return str()
        syntagmexpressions = [synt.express(False, showcoart) for synt in self.synts]
        res = "[" + " | ".join(syntagmexpressions) + "]"
        if onlytextual:
            res = res.replace("~", "NNN").replace(" | ", "-") # o~: oNNN, on: on
            res = "".join(re.findall("[0-9A-Za-z-]+", res))
        if limit != None and len(res) > limit:
            if limit > 5:
                cut = res[:limit-5]
                cut = cut if not cut.endswith("-") else cut[:-1]
                res = cut + "(...)"
            else:
                res = res[:limit]
        if protectcurlybraces:
            res = res.replace("{", "{{").replace("}", "}}")
        if vocal:
            print res
        else:
            return res


    def how_long(self, t0=2000, pause=40, iterstep=10,
                      coartmode="COMPLEX", speechrate="Normal", copygridfrom="OFF"):
        """Calculates how long the utterance is going to take.
        Input:
            t0: integer, the moment when utterance production should begin,
                in ms. By default, t0 is 2000 [ms].
            pause: integer, number of milliseconds for a pause between syntagms
                By default, pause is 40 [ms].
            iterstep: integer. A new articulatory vector is formed at least
                every iterstep ms. By default, iterstep is 10 [ms].
            coartmode: string, the mode of the coarticulation model.
                Expected values:
                - "LIN": linear transition between target vectors;
                - "COS": cosine (smoother) transition between target vectors;
                - "COMPLEX": cosine transition with finer operation of
                    articulators.
                By default, it is "COMPLEX".
                Currently it is a dummy argument: the explored time moments
                are the same for any coarticulation mode.
            speechrate: string, a dummy argument for regulating speech rate.
                Currently, all speech is synthesized at a normal rate.
            copygridfrom: string. Regulates whether the control should be
                imported from an external file. By default, no ("OFF");
                otherwise, copygridfrom is the folder containing the control files.
        Output:
            integer: the total duration of the utterance, in ms.
        """
        grid, d, v, addr = self.temporal_grid(t0, pause,
                                            iterstep, coartmode, speechrate,
                                            copygridfrom)
        return grid[-1] - grid[0]
 
   
    def temporal_grid(self, t0=2000, pause=40, iterstep=10,
                      coartmode="COMPLEX", speechrate="Normal",
                      copygridfrom="OFF"):
        """Determines what are the moments in time when, according to given
            iteration step, coarticulation mode and speech rate, vocal tract
            configuration samples are going to be generated.
        Input:
            t0: integer, the moment when utterance production should begin,
                in ms. By default, t0 is 2000 [ms].
            pause: integer, number of milliseconds for a pause between syntagms
                By default, pause is 40 [ms].
            iterstep: integer. A new articulatory vector is formed at least
                every iterstep ms. By default, iterstep is 10 [ms].
            coartmode: string, the mode of the coarticulation model.
                Expected values:
                - "LIN": linear transition between target vectors;
                - "COS": cosine (smoother) transition between target vectors;
                - "COMPLEX": cosine transition with finer operation of
                    articulators.
                By default, it is "COMPLEX".
                Currently it is a dummy argument: the explored time moments
                are the same for any coarticulation mode.
            speechrate: string, a dummy argument for regulating speech rate.
                Currently, all speech is synthesized at a normal rate.
            copygridfrom: string. Regulates whether the temporal control should be
                imported from an external file. By default, no ("OFF");
                otherwise, copygridfrom is the folder containing the control files.
        Output:
            grid: list of moments in time, in ms, when we set an articulatory
                configuration to occur.
            griddecipher: list of strings, where all elements correspond to
                the elements in grid and explain what is happening at this
                moment (see the table below).
            gridvoicing: list of boolean values: whether or not vocal folds
                vibrate at a particular moment (which one, is determined from
                the corresponding element in grid).
                It is necessary to store it separately, not take it from the
                phoneme itself, because phonemes are not necessarily pronounced
                with voice during all their production.
            gridaddresses: list of tuples (q, n, k), where all elements are
                integers and indicate the phoneme in production in the
                following way:
                    self.synts[q].sylls[n].constituents[k]
                If currently nothing is being produced, the tuple is replaced
                by "#".
                So, gridaddresses may look like
                ["#", "#", (0,0,0), (0,0,1), (0,0,2), (0,1,0), "#", "#"...]
                
            Elements of griddecipher:
                    
                    *   "#"     Silence
                    ___________________________________________________________
                    
                    *   "o"+... The vocal tract is open. Producing an open
                                vowel.
                    *   "mo"+...The vocal tract is mid-open. Producing an
                                open-mid vowel.
                    *   "mc"+...The vocal tract is mid-close. Producing a
                                close-mid vowel.
                    *   "c"+... Producing a closed vowel.
                    *   "v"+... The vocal tract is open, but it is not
                                specified how much.
                    ___________________________________________________________
                    
                    *   "A"+... Articulators are positioned for an approximant
                                or for a semivowel.
                    *   "F"+... Articulators are positioned for a fricative.
                    *   "S"+... Articulators are positioned for a stop.
                    *   "L"+... Articulators are positioned for a liquid
                                consonant.
                    *   "N"+... Articulators are positioned for a nasal
                                consonant.
                    *   "C"+... Articulators are positioned for a consonant,
                                but it is not specified of which kind.
                    ___________________________________________________________
                    
                    Additionally, there are signs for different stages of
                    phoneme production:
                    
                    *   "->"    Reaching for the phoneme target position if
                                positioned before the phoneme class sign, and
                                transitioning from there if it comes after it:
                                -   "->o":  transitioning to the target for an
                                            open vowel;
                                -   "mc->": transitioning from the target
                                            position;
                                -   "->F":  articulators are approaching each
                                            other to make constriction required
                                            for a fricative;
                                -   "S->":  after the burst phase of a stop.
                    *   "*"     The burst phase of a stop: "S*".
                    *   "!"     Being in the target position, e.g.:
                                -   "c!":   production of a closed vowel when
                                            the target configuration has been
                                            reached;
                                -   "S!":   the hold phase of a stop.
                    *   "~!"    Being near the target position but not
                                necessarily in it. It is expected to be used
                                in liquids, semivowels, approximants:
                                -   "A~!"   - the resulting sequence for an
                                            approximant may be "->A", "->A",
                                            "A~!", "A~!", "A!", "A~!", "A->".
                    ___________________________________________________________                 
        """
        if copygridfrom == "OFF":
            turnpt = 65
            grid = range(t0 - int(0.5*pause), t0, iterstep)
            griddecipher = ["#"]*len(grid)
            gridvoicing = [False]*len(grid)
            gridaddresses = ["#"]*len(grid)
            prevwassilent = True
            marker = {"Open": "o", "Open-mid": "mo", "Close-mid": "mc", "Close": "c",
                      "Stop": "S", "Fricative": "F", "Affricate": "F",
                      "Semivowel": "A", "Approximant": "A",
                      "Liquid": "L", "Nasal": "N"}
            prev = t0
            for q, synt in enumerate(self.synts):
                for n, syll in enumerate(synt.sylls):
                    for k, ph in enumerate(syll.constituents):
                        newpart = range(prev, prev + ph.duration, iterstep)
                        locturnpt = turnpt if ph.duration > turnpt else 0.48*ph.duration
                        if ph.phclass == "V":
                            try:
                                mark = marker[ph.artfeatures[0]]
                            except:
                                mark = "v"
                            # We set that it takes a vowel 100 ms to reach its
                            # target position:
                            if not prevwassilent:
                                ult = prev + min(ph.duration-locturnpt, 50) # This will define the target moment: either at prev+100, or at the latest moment possible, at the final moment minus the turning time
                            else:
                                ult = prev
                            if ult not in newpart:
                                newpart.append(ult)
                            if prev+ph.duration-locturnpt not in newpart: # The moment to start the transition to the next phoneme.
                                newpart.append(prev+ph.duration-locturnpt)
                            newpart.sort()
                            nv = [True]*len(newpart)
                            targmoment = newpart.index(ult)
                            leavmoment = newpart.index(prev+ph.duration-locturnpt)
                            nd = ["->"+mark]*len(newpart[:targmoment])
                            nd += [mark+"!"]*len(newpart[targmoment:leavmoment])
                            nd += [mark+"->"]*len(newpart[leavmoment:])
                            prevwassilent = False
                        elif ph.phclass == "C":
                            try:
                                mark = marker[ph.artmanner]
                            except:
                                mark = "C"
                                print ph.artmanner+"s are not handled! ([" + \
                                                                    ph.name + "])"
                            if mark == "S" or mark == "N":
                                # There are three steps in producing a stop:
                                #   * Catch: the articulators come into contact,
                                #       blocking the way for the air
                                #   * Hold: the articulators stay in contact
                                #   * Burst: the built-up pressure is released
                                #       through lifting the constriction.
                                # This temporal control is of a similar nature to
                                # the one of fricatives, even though the physical
                                # behaviour when producing a fricative sound is
                                # completely different. The same goes for nasals.
                                # So, right now groups "S", "N", and "F" are
                                # treated together, but it may be changed later.
                                # if syll.lengths[k] == "Long":
                                #     # 2.7 is the lengthening coefficient
                                #     # introduced in the Syllable class
                                #     # catch, hold, burst
                                #     catch = prev    # + (int(ph.duration*0.045/2.7) if not prevwassilent else 0)
                                #     burst = prev + int(ph.duration*(1-0.045/2.7))
                                #     # hold = int(3*prev + ph.duration - catch - burst)
                                # else:
                                #     # catch, hold, burst
                                catch = prev        # + (int(ph.duration*0.045) if not prevwassilent else 0) # p = 100ms => 5 ms
                                normbursttime = prev + ph.duration - 30
                                burst = normbursttime if normbursttime > prev else prev + int(ph.duration*0.6) # 5 ms
                                # hold = int(3*prev + ph.duration - catch - burst)
                                for mom in [catch, burst]: # [catch, hold, burst]:
                                    if mom not in newpart:
                                        newpart.append(mom)
                                newpart.sort()
                                catchm = newpart.index(catch)
                                # holdm = newpart.index(hold)
                                burstm = newpart.index(burst)
                                nd = ["->"+mark]*len(newpart[:catchm])
                                # nd += [mark+"!"]*len(newpart[catchm:holdm])
                                nd += [mark+"!"]*len(newpart[catchm:burstm])
                                # nd += [mark+"*"]*len(newpart[holdm:burstm])
                                # nd += [mark+"->"]*len(newpart[burstm:])
                                nd += [mark+"*"]
                                nd += [mark+"->"]*(len(newpart[burstm:])-1)
                                if ph.voicing: # Voiced stop
                                    voiceonset = int(0.5*(catchm+burstm))
                                    if k >= 1:
                                        prevph = syll.constituents[k-1]
                                    elif n >= 1:
                                        prevph = synt.sylls[n-1].constituents[-1]
                                    else:
                                        prevph = None
                                    if prevph == None or not prevph.voicing:
                                        nv = [False]*len(newpart[:voiceonset])
                                    else:
                                        nv = [True]*len(newpart[:voiceonset])
                                    nv += [True]*len(newpart[voiceonset:])
                                else: # Voiceless stop
                                    nv = [False]*len(newpart[:burstm])
                                    try:
                                        nextph = syll.constituents[k+1]
                                    except: # Last phoneme in a syllable
                                        try:
                                            nextph = synt.sylls[n+1].constituents[0]
                                        except: # Last phoneme in a syntagm
                                            nextph = None
                                    if nextph == None or not nextph.voicing:
                                        nv += [False]*len(newpart[burstm:])
                                    else: # The next phoneme is voiced
                                        nv += [True]*len(newpart[burstm:])    
                            elif mark == "F" or mark == "C":
                                nd = [mark+"!"]*len(newpart)
                                nv = [ph.voicing]*len(newpart)
                            else: # mark = "A": approximant-style constriction
                                # (an approximant, semivowel, or a liquid
                                # consonant - but any case can be singled out if
                                # necessary)
                                # The target position is passed on the way
                                # but the vectors temporally around the target
                                # configuration are rather close to the target
                                if not prevwassilent:
                                    nearult = prev + int(0.3*ph.duration)
                                    ult = prev + int(0.5*ph.duration)
                                else:
                                    nearult = ult = prev
                                afterult = prev + int(0.6*ph.duration)    
                                leav = prev + int(0.95*ph.duration)
                                for mom in [nearult, ult, afterult, leav]:
                                    if mom not in newpart:
                                        newpart.append(mom)
                                newpart.sort()
                                nv = [ph.voicing]*len(newpart)
                                neartargm = newpart.index(nearult)
                                targm = newpart.index(ult)
                                aftertargm = newpart.index(afterult)
                                leavm = newpart.index(leav)
                                nd = ["->"+mark]*len(newpart[:neartargm])
                                nd += [mark+"~!"]*len(newpart[neartargm:targm])
                                nd += [mark+"!"]*len(newpart[targm:aftertargm])
                                nd += [mark+"~!"]*len(newpart[aftertargm:leavm])
                                nd += [mark+"->"]*len(newpart[leavm:])
                            prevwassilent = False
                        elif ph.phclass == "SILENCE":
                            prevwassilent = True
                        grid.extend(newpart)
                        griddecipher.extend(nd)
                        gridvoicing.extend(nv)
                        gridaddresses.extend([(q, n, k)]*len(newpart))
                        prev += ph.duration
                # End syntagm, connect in to the next one or to the end by a pause:
                newpart = range(prev, prev+pause, iterstep)
                nd = ["#"]*len(newpart)
                nv = [False]*len(newpart)
                grid.extend(newpart)
                griddecipher.extend(nd)
                gridvoicing.extend(nv)
                gridaddresses.extend(["#"]*len(newpart))
                prev += pause
            return grid, griddecipher, gridvoicing, gridaddresses
        # copygridfrom contains the folder with temporal control files
        # Warning: the value of the limit parameter potentially different:
        timecontrolfile = self.express(False, False, True, 30, True) + ".uttx"
        # ".uttx" files are assumed to be written by human.
        # They do not respect the condition that we expect samples at least
        # every iterstep ms, so we will have to expand the time grid.
        # Control file sections like this:
        #       2030.   [a]    ->o    (0, 0, 0)
        #       2052.   [a]    o!    (0, 0, 0)
        # are interpreted as follows:
        #       2030.   [a]    ->o    (0, 0, 0)
        #       2040.   [a]    ->o    (0, 0, 0)
        #       2050.   [a]    ->o    (0, 0, 0)
        #       2052.   [a]    o!    (0, 0, 0)
        grid = list()
        griddecipher = list()
        gridvoicing = list()
        gridaddresses = list()
        addresses = [(q, n, k) for q, synt in enumerate(self.synts) for n, syll in enumerate(synt.sylls) for k, ph in enumerate(syll.constituents)]
        prevph = prevmark = prevvoice = prevaddressidx = addressidx = None
        with open(os.path.join(copygridfrom, timecontrolfile), "r") as timecontrol:
            instructions = timecontrol.readlines()
            # delay = int(instructions[0][6:])        # The first line is about the delay time: "Delay 1932"
            for line in instructions[1:]:
                instrparams = line.split()
                t = int(instrparams[0][:-1]) # + delay      # FOR THE CURRENT PROGRAM RUN
                ph = instrparams[1].replace("[", "").replace("]", "")
                mark = instrparams[2] if ph != "#" else "#"
                voice = Phoneme(ph).voicing if ph != "#" else False
                if ph == "#":
                    address = "#"
                else:
                    if prevaddressidx == None:
                        addressidx = 0
                    elif prevph != ph or (prevph == ph and prevmark.endswith("->") and mark.startswith("->")):
                        addressidx += 1
                    address = addresses[addressidx]   
                if len(grid) > 0:
                    if t - grid[-1] > iterstep:
                        missedmoms = range(grid[-1] + iterstep, t, iterstep)
                        grid.extend(missedmoms)
                        griddecipher.extend([griddecipher[-1]]*len(missedmoms))
                        gridvoicing.extend([gridvoicing[-1]]*len(missedmoms))
                        gridaddresses.extend([gridaddresses[-1]]*len(missedmoms))
                grid.append(t)
                griddecipher.append(mark)
                gridvoicing.append(voice)
                gridaddresses.append(address)
                prevph, prevmark, prevvoice, prevaddressidx = ph, mark, voice, addressidx
        return grid, griddecipher, gridvoicing, gridaddresses
    

    def get_utt_boundaries(self, t0=2000, pause=40, iterstep=10,
                      coartmode="COMPLEX", speechrate="Normal",
                      copygridfrom="OFF"):
        # We need to determine tfirst and tlast: the moments of time beyond which
        # nothing is uttered.
        g, d, v, a = self.temporal_grid(t0, pause, iterstep, coartmode,
                                        speechrate, copygridfrom)
        tfirst, tlast = t0, g[-1]
        for k in range(len(g)-2, -1, -1):
            if d[k+1]=="#":
                tlast = g[k]
            else:
                break
        for k, t in enumerate(g):
            if t < t0:
                continue
            # Warning: implicit usage of the limited options in the temporal control.
            if d[k] != "#":
                tfirst = g[k]
                break
        return tfirst, tlast


    def how_many_samples(self, t0=2000, pause=40, iterstep=10,
                         coartmode="COMPLEX", speechrate="Normal", copygridfrom="OFF"):
        """Calculates how many iterations in vocal tract configuration it will
        take to synthesize the given utterance.
        Input:
            t0: integer, the moment when utterance production should begin,
                in ms. By default, t0 is 1200 [ms].
            pause: integer, number of milliseconds for a pause between syntagms
                By default, pause is 40 [ms].
            iterstep: integer. A new articulatory vector is formed at least
                every iterstep ms. By default, iterstep is 10 [ms].
            coartmode: string, the mode of the coarticulation model.
                Expected values:
                - "LIN": linear transition between target vectors;
                - "COS": cosine (smoother) transition between target vectors;
                - "COMPLEX": cosine transition with finer operation of
                    articulators.
                By default, it is "COMPLEX".
            speechrate: string, a dummy argument for regulating speech rate.
                Currently, all speech is synthesized at a normal rate.
            copygridfrom: string. Regulates whether the control should be
                imported from an external file. By default, no ("OFF");
                otherwise, copygridfrom is the folder containing the control files.
        Output:
            integer: the total number of samples for the utterance, in ms.
        """
        return len(self.temporal_grid(t0, iterstep, coartmode, speechrate, copygridfrom)[0])

    
    def record_art_vectors(self, artvectfile="Syntheses/{}_{}_data/{}_{}.vtp",
                           t0=2000, pause=40, iterstep=10, vocal=True,
                           coartmode="COMPLEX", speechrate="Normal", copygridfrom="OFF"):
        """Records the articulatory vector sequence for the utterance.
        Input:
            artvectfile: string, the path for the output file.
                By default, it is set as "Syntheses/{}_{}_data/{}_{}.vtp".
            t0: integer, the moment when utterance production should begin,
                in ms. By default, t0 is 2000 [ms].
            pause: integer, number of milliseconds for a pause between syntagms
                By default, pause is 40 [ms].
            iterstep: integer. A new articulatory vector is formed at least
                every iterstep ms. By default, iterstep is 10 [ms].
            vocal: boolean. Whether to print the status of the function when it
                finishes. By default, yes (True).
            coartmode: string, the mode of the coarticulation model.
                Expected values:
                - "LIN": linear transition between target vectors;
                - "COS": cosine (smoother) transition between target vectors;
                - "COMPLEX": cosine transition with finer operation of
                    articulators.
                By default, it is "COMPLEX".
            speechrate: string, a dummy argument for regulating speech rate.
                Currently, all speech is synthesized at a normal rate.
            copygridfrom: string. Regulates whether the control should be
                imported from an external file. By default, no ("OFF");
                otherwise, copygridfrom is the folder containing the control files.
        Output:
            status: string describing what has been done if vocal is True;
                otherwise None.
        """
        if self.status == "ERROR":
            return
        if artvectfile.endswith("{}_{}_data/{}_{}.vtp"):
            slug = self.express(False, False, True, 30, True)
            artvectfile = artvectfile.format(slug, coartmode, slug, coartmode)
        foldername = artvectfile[:artvectfile.rfind("/")]
        filename = artvectfile[artvectfile.rfind("/")+1:]
        where = foldername[:foldername.rfind("/")]
        which = foldername[foldername.rfind("/")+1:]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            status = "The folder \""+which+"/\" has been created in "
            status += "\""+where+"/\".\n"
        else:
            status = str()
        vt = VocalTract()
        lines = [str(vt.totalparams)]
        grid, decipher, voicing, phenumeration = self.temporal_grid(t0, pause, \
                iterstep, coartmode, speechrate, copygridfrom)
        targetmoments = [grid[0]]
        artvectors = [Phoneme("silence").artv["Solo"]]
        # Warning: Hidden usage of the default Phoneme constructor arguments!
        criticalarts = [Phoneme("silence").critart]
        for t, mark, prevmark, address, prevaddress \
                in zip(grid, decipher, ["#"]+decipher,
                        phenumeration, ["#"]+phenumeration):
            if ("!" in mark and "~!" not in mark) or \
                    (prevmark.startswith("->") and mark.endswith("->")):
                q, n, k = address
                anticipation = self.synts[q].anticipations[n][k]
                ph = self.synts[q].sylls[n].constituents[k]
                targetmoments.append(t)
                artvectors.append(ph.artv[anticipation])
                criticalarts.append(ph.critart)
        targetmoments.append(grid[-1])
        artvectors.append(Phoneme("silence").artv["Solo"])
        criticalarts.append(Phoneme("silence").critart)
        # Warning: Hidden usage of the default Phoneme constructor arguments!
        slug = self.express(False, False, True, 30, True) # Just for logging purposes, can be removed when the algorithm is ready
        for v in interpolate(artvectors, targetmoments, grid, t0, \
                            phenumeration, decipher, criticalarts, VocalTract(), slug, coartmode):
            lines.append(" ".join(['{0:.2f}'.format(el) for el in v]))
        artvtext = "\n".join(lines)
        with open(artvectfile, "w") as avf:
            avf.write(artvtext)
            avf.close()
        if status:
            status += "There, file \"{}\" listing the articul".format(filename)
            status += "atory vectors for the utterance has been created.\n"
        else:
            status += "File \"{}\" listing the articulatory ".format(filename)
            status += "vectors for the utterance has been created in the "
            status += "\"{}/\" folder in \"{}/\".\n".format(which, where)
        if vocal:
            print status
        else:
            return status


    def record_af_list(self, affile="Syntheses/{}_{}_data/{}_{}.af",
                       artvfile="Syntheses/{}_{}_data/{}_{}.vtp",
                       uniqartvfile="Syntheses/{}_{}_data/{}_{}.vtpx",
                       t0=2000, pause=40, iterstep=10, vocal=True,
                       coartmode="COMPLEX", speechrate="Normal",
                       copygridfrom="OFF", corpuslogger=str()):
        """Records the area function file sequence for the utterance.
        Input:
            affile: string, the path for the output file.
                By default, it is set as "Syntheses/{}_{}_data/{}_{}.af".
            t0: integer, the moment when utterance production should begin,
                in ms. By default, t0 is 2000 [ms].
            pause: integer, number of milliseconds for a pause between syntagms
                By default, pause is 40 [ms].
            iterstep: integer. A new articulatory vector is formed at least
                every iterstep ms. By default, iterstep is 10 [ms].
            vocal: boolean. Whether to print the status of the function when it
                finishes. By default, yes (True).
            coartmode: string, the mode of the coarticulation model.
                Expected values:
                - "LIN": linear transition between target vectors;
                - "COS": cosine (smoother) transition between target vectors;
                - "COMPLEX": cosine transition with finer operation of
                    articulators.
                By default, it is "COMPLEX".
            speechrate: string, a dummy argument for regulating speech rate.
                Currently, all speech is synthesized at a normal rate.
            copygridfrom: string. Regulates whether the control should be
                imported from an external file. By default, no ("OFF");
                otherwise, copygridfrom is the folder containing the control files.
        Output:
            status: string describing what has been done if vocal is True;
                otherwise None.
            Writes an *.af file containing the sequence of area functions
                involved
        NB! The assumed name for *.xa files is e.g. "AF00000.xa", i.e.
            "AF" + id padded with zeros to the left to the length of 5 + ".xa":
                "202
                40
        
                1300.
                AF00000.xax
                ***
        
                1310.
                AF00001.xax
                COS
        
                1320.
                AF00002.xax
                COS"
        """
        if self.status == "ERROR":
            return
        if affile.endswith("{}_{}_data/{}_{}.af"):
            slug = self.express(False, False, True, 30, True)
            affile = affile.format(slug, coartmode, slug, coartmode)
        if artvfile.endswith("{}_{}_data/{}_{}.vtp"):
            slug = self.express(False, False, True, 30, True)
            artvfile = artvfile.format(slug, coartmode, slug, coartmode)
        if uniqartvfile.endswith("{}_{}_data/{}_{}.vtpx"):
            slug = self.express(False, False, True, 30, True)
            uniqartvfile = uniqartvfile.format(slug, coartmode, slug, coartmode)
        foldername = affile[:affile.rfind("/")]
        filename = affile[affile.rfind("/")+1:]
        where = foldername[:foldername.rfind("/")]
        which = foldername[foldername.rfind("/")+1:]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            status = "The folder \""+which+"/\" has been created in "
            status += "\""+where+"/\".\n"
        else:
            status = str()
        g, d, v, addr = self.temporal_grid(t0, pause, iterstep, coartmode,
                                        speechrate, copygridfrom)
        tfirst, tlast = self.get_utt_boundaries(t0, pause, iterstep, coartmode,
                                        speechrate, copygridfrom)
        afindexer = dict()
        with open(artvfile) as vtpfile:
            allartvectors = vtpfile.readlines()
            uniqartvlist = [allartvectors[0]]
            # WARNING! ONLY FOR THIS RUN! NEXT TIME THE INDEXES OF AFs AND IN .AF WILL BE THE SAME! - Leaving the comment, but it doesn't seem to be true now.
            for k, vect in enumerate(allartvectors[1:]):
                if g[k]<tfirst or g[k]>tlast or d[k]=="#":
                # if g[k]<t0 or g[k]>tlast:
                    continue
                try:
                    uniqnumber = uniqartvlist.index(vect) - 1   # Because of the number of parameters in the first line.
                except:
                    uniqartvlist.append(vect)
                    uniqnumber = len(uniqartvlist) - 2          # Because of the number of parameters in the first line.
                afindexer[k] = uniqnumber
            if corpuslogger:
                overallvtpx = corpuslogger[:corpuslogger.rfind(".")] + ".vtpc"
                corpindexer = dict()
                curraddition = uniqartvlist[1:]
                uniqpartofaddition = list()
                if os.path.exists(corpuslogger):
                    with open(corpuslogger, "r") as corplog:
                        corpentryid = int(corplog.readlines()[-1])
                else:
                    corpentryid = 0
                with open(overallvtpx, "a+") as corpvtpx:
                    corpusvectors = corpvtpx.readlines()
                    oldlen = len(corpusvectors)
                    for k, vect in enumerate(curraddition):
                        try:
                            uniqnumber = corpusvectors.index(vect) - 1 
                        except:
                            uniqpartofaddition.append(vect)
                            uniqnumber = (oldlen-1 if oldlen else oldlen) + len(uniqpartofaddition) - 1
                        corpindexer[k] = uniqnumber
                    if oldlen == 0:
                        corpvtpx.write(str(VocalTract().totalparams)+"\n")
                    corpvtpx.write("".join(uniqpartofaddition))
                # corp2locidxer = {v: k for k, v in corpindexer.iteritems()}
                with open(corpuslogger, "a+") as corplog:
                    newlog = "{}_{}\n".format(self.express(False, False, True, 30, True), coartmode)
                    localdir = affile[:affile.rfind("/") + 1]
                    corpusdir = corpuslogger[:corpuslogger.rfind("/") + 1]
                    for k, _ in enumerate(curraddition):
                        wheretoputtheAFs = overallvtpx[:overallvtpx.rfind(".")] + "_data/"
                        newlog += "{}AF{}.xax --> {}AF{}.xax\n".format(wheretoputtheAFs, format(corpindexer[k], 5), localdir, format(k, 5))
                        # newlog += "{}AF{}.xax --> {}AF{}.xax\n".format(wheretoputtheAFs, format(corpindexer[k], 5), localdir, format(k, 5))
                        # newlog += "AF{}_{} = AF{}\n".format(format(corpentryid, 5), format(k, 5), format(corpindexer[k], 5))
                    newlog += "------------------------------------------\n"
                    newlog += str(corpentryid + 1) + "\n"
                    corplog.write(newlog)
        uniqartvtext = "".join(uniqartvlist)
        with open(uniqartvfile, "w") as uniqartvf:
            uniqartvf.write(uniqartvtext)
        lines = list() 
        for afid, t in enumerate(g):
            if t<tfirst or t>tlast or d[afid]=="#":
                continue
                # We do not cut g[g.index(t0):g.index(tlast)+1] in order to
                # preserve the numeration in area function files.
            join = "COS" if len(lines) != 0 else "***"
            lines.append("{}.\nAF{}.xax\n{}".format(t, format(afindexer[afid], 5), join))
        norepetlines = [lines[0]]
        for line, prevline, nextline in zip(lines[1:], lines, lines[2:]):
            if line[-11:] == prevline[-11:] and line[-11:] == nextline[-11:]:
                continue    # This line does not add any additional information.
            norepetlines.append(line)
        norepetlines.append(lines[-1])
        # The number of tubes is set to 40:
        aftext = str(len(norepetlines))+"\n40\n\n" + "\n\n".join(norepetlines)
        with open(affile, "w") as af:
            af.write(aftext)
            af.close()
        if status:
            status += "There, file \"{}\" listing the proper".format(filename)
            status += " area function files has been created.\n"
        else:
            status += "File \"{}\" listing the proper area ".format(filename)
            status += "function files has been created in the "
            status += "\"{}/\" folder in \"{}/\".\n".format(which, where)
        if vocal:
            print status
        else:
            return status

    
    def record_glottal_opening(self, glpressfile=\
                                   "Syntheses/{}_{}_data/{}_{}.ag0",
                                    t0=2000, pause=40, iterstep=10, vocal=True,
                                    coartmode="COMPLEX", speechrate="Normal",
                                    copymatfrom="OFF", copygridfrom="OFF"):
        """Records the glottal opening changes for the utterance.
        Input:
            glpressfile: string, the path for the output file.
                By default, it is set as "Syntheses/{}_{}_data/{}_{}.ag0".
            t0: integer, the moment when utterance production should begin,
                in ms. By default, t0 is 2000 [ms].
            pause: integer, number of milliseconds for a pause between syntagms
                By default, pause is 40 [ms].
            iterstep: integer. A new articulatory vector is formed at least
                every iterstep ms. By default, iterstep is 10 [ms].
            vocal: boolean. Whether to print the status of the function when it
                finishes. By default, yes (True).
            coartmode: string, the mode of the coarticulation model.
                Expected values:
                - "LIN": linear transition between target vectors;
                - "COS": cosine (smoother) transition between target vectors;
                - "COMPLEX": cosine transition with finer operation of
                    articulators.
                By default, it is "COMPLEX".
            speechrate: string, a dummy argument for regulating speech rate.
                Currently, all speech is synthesized at a normal rate.
            copymatfrom: string. Regulates whether the glottal opening data should be
                imported from an external file. By default, no ("OFF");
                otherwise, copymatfrom is the folder containing the control files.
            copygridfrom: string. Regulates whether the glottal opening data should be
                imported from an external file. By default, no ("OFF");
                otherwise, copygridfrom is the folder containing the control files.    
        Output:
            status: string describing what has been done if vocal is True;
                otherwise None.
            Writes an *.ag0 file that controls glottal opening.
                "49
                1326. 0.4 ***
                1419. 0.4 LIN
                1439. 0 COS
                1473. 0 LIN
                1490. 0 LIN
                (...)"
        """
        if self.status == "ERROR":
            return
        if glpressfile.endswith("{}_{}_data/{}_{}.ag0"):
            slug = self.express(False, False, True, 30, True)
            glpressfile = glpressfile.format(slug, coartmode, slug, coartmode)
        foldername = glpressfile[:glpressfile.rfind("/")]
        filename = glpressfile[glpressfile.rfind("/")+1:]
        where = foldername[:foldername.rfind("/")]
        which = foldername[foldername.rfind("/")+1:]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            status = "The folder \""+which+"/\" has been created in "
            status += "\""+where+"/\".\n"
        else:
            status = str()
        g, d, v, addr = self.temporal_grid(t0, pause, iterstep, coartmode,
                                        speechrate, copygridfrom)
        tfirst, tlast = self.get_utt_boundaries(t0, pause, iterstep, coartmode,
                                                speechrate, copygridfrom)
        # lines = ["{}. 1.0 ***".format(tfirst)]
        lines = list()
        # fin = "{}. 1.0 LIN".format(tlast)
        if copymatfrom == "OFF":
            # The glottis opens for obstruents (pressure consonants):
            # stops, fricatives, and affricates.
            # The degree of opening differs across different sounds, for example,
            # for voiced fricatives, the glottis is half-closed, half-open. 
            dfactorized = list()
            # # Commented out on March 10, 2017 because ag0 now controls the glottal opening as a relative value
            # for el, prevel, voice in zip(d, ["#"]+d, v):      
            #     if el == "S!" or (el == "S*" and prevel == "S!") or ("F" in el and not voice):
            #         dfactorized.append("Obstruent")
            #     elif "F" in el and voice:
            #         dfactorized.append("~Obstruent")
            #     else:
            #         dfactorized.append("Sonorant")
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # [[tshift, vv1st/1.2], [tv1st, vv1st], [tv1end, vv1end], [tpeakst, vpeakst], [tpeak, vpeak], [tpeakend, vpeakend], [tv2st, vv2st], [tv2end, vv2end], [tv2end+100, vv2end/1.2]]
            # Voiced:
            # [1937.0 -> 0.085; 1991.4 -> 0.102; 2081.3 -> 0.399; 2141.3 -> 0.613; 2191.3 -> 0.699; 2241.3 -> 0.593; 2341.3 -> 0.170; 2441.3 -> 0.072; 2541.3 -> 0.060]
            # Unvoiced:
            # [1937.0 -> 0.180; 1995.2 -> 0.215; 2085.2 -> 0.423; 2145.2 -> 0.784; 2195.2 -> 1.000; 2245.2 -> 0.871; 2345.2 -> 0.220; 2445.2 -> 0.069; 2545.2 -> 0.058]
            # First, identify peaks.
            initialknots = [[g[0], "#"]]
            newpeak = None
            newvowel = None
            # phraserepr = str()
            for el, prevel, nextel, voice, t in zip(d, ["#"]+d, d[1:]+["#"], v, g):
                # if re.sub(r'\W+', '', el) != re.sub(r'\W+', '', prevel):
                #     phraserepr += re.sub(r'\W+', '', el) if el != "#" else "#"
                wasvowel = any(marker in prevel for marker in ["o", "mo", "mc", "c", "v"])
                isvowel = any(marker in el for marker in ["o", "mo", "mc", "c", "v"])
                willbevowel = any(marker in nextel for marker in ["o", "mo", "mc", "c", "v"])
                if isvowel and not wasvowel:
                    newvowel = t
                    initialknots.append([newvowel, False])
                if isvowel and not willbevowel:
                    # initialknots.append([int(newvowel + 0.4*(t - newvowel)), False])
                    initialknots.append([int(newvowel + 0.5*(t - newvowel)), False])
                if isvowel:
                    continue
                # if el != nextel:
                #     if initialknots:
                #         if initialknots[-1][1] != False:
                #             initialknots.append([int(newvowel + 0.24*(t - newvowel)), False])
                #     else:
                #         initialknots.append([int(newvowel + 0.24*(t - newvowel)), False])
                #     continue
                # if el == prevel and (any(marker in el for marker in ["o!", "mo!", "mc!", "c!", "v!"]) or (any(marker in el for marker in ["->o", "->mo", "->mc", "->c", "->v"]) and any(marker in el for marker in ["o->", "mo->", "mc->", "c->", "v->"]))):
                #     continue
                if el == "#" and (prevel != "#" or nextel != "#"):
                    initialknots.append([t, "#"])
                    continue
                if "~!" not in el and "!" in el :
                    if el != prevel:
                        newpeak = t
                    if el != nextel:
                        # Peak between newpeak and t.
                        peakt = int(newpeak + 0.45*(t-newpeak))
                        if "F" in el or "S" in el:
                            if not voice:
                                peakval = 1.0
                            else:
                                peakval = 0.699
                        elif any(marker in el for marker in ["A", "L", "N", "C"]): # L - 0, R - ?, N - 0
                            peakval = False
                            # if not voice:
                            #     peakval = 0.8
                            # else:
                            #     peakval = 0.65
                        initialknots.append([peakt, peakval])
            arepeaks = [knot[1] if knot[1] in [False, "#"] else True for knot in initialknots]
            allknots = list()
            def shift(moments, lower, upper=None, margin=10):
                minval = min(moments)
                if minval > lower:
                    return moments
                shiftval = lower - minval + margin
                if upper:
                    maxval = max(moments)
                    if maxval + shiftval >= upper:
                        span = maxval - minval
                        newmaxval, newminval = upper - margin, lower + margin
                        newspan = newmaxval - newminval
                        if newspan <= 0:
                            return shift(moments, lower, upper, int(margin*1.0/2))
                        loccoords = [(minval - t)*1.0/span for t in moments]
                        return [int(newminval + alpha*newspan) for alpha in loccoords]
                    else:
                        return [t+shiftval for t in moments]
                else:
                    return [t+shiftval for t in moments]
            # Construct the lower knots around the initialknots. It is necessary only between any True and False, not between True and True.
            for (k, ispeak), willbepeak in zip(enumerate(arepeaks), arepeaks[1:]):
                tcurr, valcurr = initialknots[k]
                tnext, valnext = initialknots[k+1]
                if ispeak == willbepeak == True:
                    if len(allknots) == 0 or allknots[-1][0] < tcurr:
                        allknots.append([tcurr, valcurr])
                    else:
                        allknots.append([allknots[-1][0] + 10, valcurr])
                else:
                    transt = tnext - tcurr
                    if ispeak == True: # From a peak
                        if len(allknots) == 0 or allknots[-1][0] < tcurr:
                            allknots.append([tcurr, valcurr])
                        else:
                            allknots.append([allknots[-1][0] + 10, valcurr])
                        if abs(valcurr - 1) < 0.04:
                            # v1, v2 = 0.871, 0.220
                            v1, v2, v3, v4 = 0.871, 0.620, 0.220, 0.000
                            if transt >= 230:
                                # t1, t2 = tcurr+50, tcurr+150
                                # t1, t2, t3, t4 = tcurr+50, tcurr+130, tcurr+150, tcurr+220
                                t1, t2, t3, t4 = tcurr+50, tcurr+130, tcurr+240, tcurr+260
                            else:
                                # t1, t2, t3, t4 = int(tcurr+0.217*transt), int(tcurr+0.565*transt), int(tcurr+0.652*transt), int(tcurr+0.957*transt)
                                t1, t2, t3, t4 = int(tcurr+0.217*transt), int(tcurr+0.565*transt), int(tcurr+1.043*transt), int(tcurr+1.13*transt)
                        else:
                            # v1, v2 = 0.593, 0.170
                            v1, v2, v3, v4 = 0.593, 0.370, 0.170, 0.000
                            if transt >= 220:
                                # t1, t2 = tcurr+50, tcurr+150
                                # t1, t2, t3, t4 = tcurr+50, tcurr+130, tcurr+150, tcurr+210
                                t1, t2, t3, t4 = tcurr+50, tcurr+130, tcurr+230, tcurr+250
                            else:
                                # t1, t2, t3, t4 = int(tcurr+0.227*transt), int(tcurr+0.591*transt), int(tcurr+0.682*transt), int(tcurr+0.955*transt)
                                t1, t2, t3, t4 = int(tcurr+0.227*transt), int(tcurr+0.591*transt), int(tcurr+1.05*transt), int(tcurr+1.14*transt)
                        if len(allknots) and t1 <= allknots[-1][0]:
                            t1, t2, t3, t4 = shift([t1, t2, t3, t4], allknots[-1][0])
                        allknots.append([t1, v1])
                        allknots.append([t2, v2])
                        allknots.append([t3, v3])
                        allknots.append([t4, v4])
                    elif willbepeak == True: # To a peak
                        if transt >= 200:
                            # t1, t2, t3 = tcurr, tnext - 110, tnext - 50
                            t1, t2, t3, t4 = tcurr, tnext - 180, tnext - 80, tnext - 35
                        else:
                            # t1, t2, t3 = tcurr, int(tnext - 0.55*transt), int(tnext - 0.25*transt)
                            t1, t2, t3, t4 = tcurr, int(tnext - 0.9*transt), int(tnext - 0.4*transt), int(tnext - 0.175*transt)
                        if abs(valnext - 1) < 0.04:
                            # v1, v2, v3 = 0.215, 0.423, 0.784
                            v1, v2, v3, v4 = 0.000, 0.000, 0.423, 0.784
                        else:
                            # v1, v2, v3 = 0.102, 0.399, 0.613
                            v1, v2, v3, v4 = 0.000, 0.000, 0.399, 0.613
                        if len(allknots) and t1 <= allknots[-1][0]:
                            t1, t2, t3, t4 = shift([t1, t2, t3, t4], allknots[-1][0])
                        allknots.append([t1, v1])
                        allknots.append([t2, v2])
                        allknots.append([t3, v3])
                        allknots.append([t4, v4])
                    elif ispeak == "#":
                        if (len(allknots) and tcurr > allknots[-1][0]) or (len(allknots) == 0):
                            allknots.append([tcurr, 0.0])
                        else:
                            if allknots[-1][0] == 0.0:
                                continue
                            allknots.append([allknots[-1][0] + 5, 0.0])
                    # The case of ispeak == willbepeak == False is not necessary.
            # if allknots[-1][0] < initialknots[-1][0]:
            #     allknots.append(initialknots[-1])
            if [g[-1], 0.0] not in allknots:
                allknots.append([g[-1], 0.0])
            t = np.array([tpt for tpt, _ in allknots])
            lch = np.array([lchpt for _, lchpt in allknots])
            spl = interp.PchipInterpolator(t, lch)
            allt = np.arange(allknots[0][0], allknots[-1][0])
            lchestim = spl(allt)
            allt = allt.astype(int)
            for moment, lchval in zip(allt, lchestim):
                if moment < t0 or moment > tlast:
                    continue
                join = "COS"
                if lines == list():
                    lines = [str(moment) + ". %3f" % lchval + " ***"]
                else:
                    lines.append(str(moment) + ". %3f" % lchval + " " + join)
            # for el, prevel, voice, t in zip(d, ["#"]+d, v, g):
            #     #if t < tfirst or t > tlast:
            #     #    continue
            #     if not voice:
            #         dfactorized.append("Voiceless")
            #     elif "F" in el and voice:
            #         dfactorized.append("VoicedFricative")
            #     else:
            #         dfactorized.append("Voiced")
            # for moment, decipher, prevdecipher, nextdecipher \
            #         in zip(g, dfactorized, ["Sonorant"]+dfactorized,
            #                dfactorized[1:]+["Sonorant"])[1:]:
            #     if moment < tfirst or moment > tlast:
            #         continue
            #     # coef = "0.40" if decipher == "Obstruent" else "0.25" if decipher == "~Obstruent" else "0.00"
            #     # # Correction of March 10, 2017:
            #     # coef = "1" if decipher == "Voiceless" else "0.5" if decipher == "VoicedFricative" else "0.00"
            #     # # Correction of March 13, 2017:
            #     coef = "1.0" if decipher == "Voiceless" else "0.5" if decipher == "VoicedFricative" else "0.00"
            #     join = "COS" if decipher != prevdecipher else "LIN"
            #     if lines == list():
            #         lines = ["{}. {} {}".format(moment, coef, "***")]
            #     elif (not lines[-1].endswith(". {} {}".format(coef, join)))\
            #         or (decipher != nextdecipher):
            #         lines.append("{}. {} {}".format(moment, coef, join))
            
        else:  
            slug = self.express(False, False, True, 30, True)
            matfile = os.path.join(copymatfrom, slug + ".mat")
            try:
                uttxfile = os.path.join(copygridfrom, slug + ".uttx")
            except:
                if copygridfrom == "OFF":
                    explanation = "Error: copygridfrom == \"OFF\"."
                else:
                    explanation = "copygridfrom = " + copygridfrom
                print "Unable to process .mat files with true f0 and lch data without a time grid. " + explanation
                raise
            with open(uttxfile, "r") as uttx:
                # dataentries = uttx.readline()
                delay = uttx.readline()
                tshift = int(delay[delay.find(" ")+1:])
            observ = imp(matfile, "to", "lch", tshift)
            join = "COS"
            for moment, coef in observ:
                lines.append("{}. {} {}".format(moment, coef, join))
        #if lines[-1] != fin:
        #    lines.append(fin)
        linesmerged = [lines[0]]
        def cut_command(command):
            return command[command.find("."):]
        def same_values(command1, command2, command3):
            return cut_command(command1) == cut_command(command2) == cut_command(command3)
        for currl, nextl in zip(lines[1:], lines[2:]):
            lastl = linesmerged[-1]
            if same_values(lastl, currl, nextl):
                continue
            else:
                linesmerged.append(currl)
        linesmerged.append(lines[-1])
        subgltext = str(len(linesmerged))+"\n" + "\n".join(linesmerged)
        with open(glpressfile, "w") as subgf:
            subgf.write(subgltext)
            subgf.close()
        if status:
            status += "There, file \"{}\" operating glottal".format(filename)
            status += " pressure has been created.\n"
        else:
            status += "File \"{}\" operating glottal ".format(filename)
            status += "pressure has been created in the "
            status += "\"{}/\" folder in \"{}/\".\n".format(which, where)
        if vocal:
            print status
        else:
            return status

    
    def record_vocal_folds(self, vocfoldsfile="Syntheses/{}_{}_data/{}_{}.agp",
                       t0=2000, pause=40, iterstep=10, vocal=True,
                       coartmode="COMPLEX", speechrate="Normal",
                       copygridfrom="OFF"):
        """Records a file regulating vocal folds oscillations in the utterance.
        Input:
            vocfoldsfile: string, the path for the output file.
                By default, it is set as "Syntheses/{}_{}_data/{}_{}.agp".
            t0: integer, the moment when utterance production should begin,
                in ms. By default, t0 is 2000 [ms].
            pause: integer, number of milliseconds for a pause between syntagms
                By default, pause is 40 [ms].
            iterstep: integer. A new articulatory vector is formed at least
                every iterstep ms. By default, iterstep is 10 [ms].
            vocal: boolean. Whether to print the status of the function when it
                finishes. By default, yes (True).
            coartmode: string, the mode of the coarticulation model.
                Expected values:
                - "LIN": linear transition between target vectors;
                - "COS": cosine (smoother) transition between target vectors;
                - "COMPLEX": cosine transition with finer operation of
                    articulators.
                By default, it is "COMPLEX".
                Currently it is a dummy argument because there is no difference
                between coarticulation modes in vocal folds operation.
            speechrate: string, a dummy argument for regulating speech rate.
                Currently, all speech is synthesized at a normal rate.
            copygridfrom: string. Regulates whether the control should be
                imported from an external file. By default, no ("OFF");
                otherwise, copygridfrom is the folder containing the control files.
        Output:
            status: string describing what has been done if vocal is True;
                otherwise None.
            Writes an *.agp file that controls vocal fold oscillations:
                        "49
                        1326. 0 ***
                        1419. 0 LIN
                        1439. 0.1 COS
                        1473. 0.1 LIN
                        1490. 0.1 LIN
                        1510. 0.1 LIN
                        1530. 0.1 LIN
                        1534. 0.1 LIN
                        1583. 0.1 COS
                        1589. 0.1 LIN
                        (...)"
                (The example is taken from the actual data: VAXL.agp
                for uttering "Il a pas mal.")
        NB: It is assumed that assimilation (voiced sounds becoming voiceless
                because of the context and vice versa) has been handled BEFORE
                providing the transcription of the utterance to synthesize.
                Examples: absent [ap-'s{a~}], obturation [{oe}p-ty-ra-'sj{o~}].
        """
        if self.status == "ERROR":
            return
        if vocfoldsfile.endswith("{}_{}_data/{}_{}.agp"):
            slug = self.express(False, False, True, 30, True)
            vocfoldsfile = vocfoldsfile.format(slug, coartmode, slug, coartmode)
        foldername = vocfoldsfile[:vocfoldsfile.rfind("/")]
        filename = vocfoldsfile[vocfoldsfile.rfind("/")+1:]
        where = foldername[:foldername.rfind("/")]
        which = foldername[foldername.rfind("/")+1:]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            status = "The folder \""+which+"/\" has been created in "
            status += "\""+where+"/\".\n"
        else:
            status = str()
        g, d, v, addr = self.temporal_grid(t0, pause, iterstep,
                                        coartmode, speechrate, copygridfrom)
        lines = ["{}. 0 ***".format(g[0])]
        for moment, voice, prevvoice, nextvoice, el \
                in zip(g, v, [False]+v, v[1:]+[False], d)[1:]:
            coef = "0.1" if (voice and "F" not in el) else \
                        "0.8" if (voice and "F" in el) else "0"
            join = "COS" if voice != prevvoice else "LIN"
            if (not lines[-1].endswith(". {} {}".format(coef, join)))\
                or (voice != nextvoice):
                lines.append("{}. {} {}".format(moment, coef, join))
        vftext = str(len(lines))+"\n" + "\n".join(lines)
        with open(vocfoldsfile, "w") as vf:
            vf.write(vftext)
            vf.close()
        if status:
            status += "There, file \"{}\" ".format(filename)
            status += "regulating vocal folds oscillations has been created.\n"
        else:
            status += "File \"{}\" regulating vocal folds ".format(filename)
            status += "oscillations has been created in the "
            status += "\"{}/\" folder in \"{}/\".\n".format(which, where)
        if vocal:
            print status
        else:
            return status
    
    
    def record_intonation(self, intonfile="Syntheses/{}_{}_data/{}_{}.f0",
                       t0=2000, pause=40, iterstep=10, vocal=True,
                       coartmode="COMPLEX", speechrate="Normal",
                       copymatfrom="OFF", copygridfrom="OFF"):
        """Records the fundamental frequency for the utterance synthesis.
        Input:
            intonfile: string, the path for the output file.
                By default, it is set as "Syntheses/{}_{}_data/{}_{}.f0".
            t0: integer, the moment when utterance production should begin,
                in ms. By default, t0 is 2000 [ms].
            pause: integer, number of milliseconds for a pause between syntagms
                By default, pause is 40 [ms].
            iterstep: integer. A new articulatory vector is formed at least
                every iterstep ms. By default, iterstep is 10 [ms].
            vocal: boolean. Whether to print the status of the function when it
                finishes. By default, yes (True).
            coartmode: string, the mode of the coarticulation model.
                Expected values:
                - "LIN": linear transition between target vectors;
                - "COS": cosine (smoother) transition between target vectors;
                - "COMPLEX": cosine transition with finer operation of
                    articulators.
                By default, it is "COMPLEX".
            speechrate: string, a dummy argument for regulating speech rate.
                Currently, all speech is synthesized at a normal rate.
            copymatfrom: string. Regulates whether the control should be
                imported from an external file. By default, no ("OFF");
                otherwise, copymatfrom is the folder containing the control files.
            copygridfrom: string. Regulates whether the control should be
                imported from an external file. By default, no ("OFF");
                otherwise, copygridfrom is the folder containing the control files.
        Output:
            string describing what has been done if vocal is True;
                otherwise None.
            Writes a *.f0 file that controls voice pitch:
            time, pitch [in Hz], mode how to connect with the previous value
                        "182
                        1244 0 SET
                        1444 181 SET
                        1448 181 LIN
                        1452 179 LIN
                        1456 179 LIN
                        1460 179 LIN
                        1464 179 LIN
                        1468 180 LIN
                        1472 180 LIN
                        1476 190 LIN
                        (...)
                        2500 170 LIN
                        2505 0 SET"
                (The example taken from the actual data: VAXL.f0 for uttering
                "Il a pas mal.")
        The function needs refactoring (inefficient parsing).
        """
        if self.status == "ERROR":
            return
        if intonfile.endswith("{}_{}_data/{}_{}.f0"):
            slug = self.express(False, False, True, 30, True)
            intonfile = intonfile.format(slug, coartmode, slug, coartmode)
        foldername = intonfile[:intonfile.rfind("/")]
        filename = intonfile[intonfile.rfind("/")+1:]
        where = foldername[:foldername.rfind("/")]
        which = foldername[foldername.rfind("/")+1:]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            status = "The folder \""+which+"/\" has been created in "
            status += "\""+where+"/\".\n"
        else:
            status = str()
        g, d, v, addr = self.temporal_grid(t0, pause, iterstep, coartmode,
                                        speechrate, copygridfrom)
        lines = []
        if copymatfrom == "OFF":
            lines = ["{} 100 SET".format(g[1])] # MANIPULATIONS WITH F0
            # The fundamental frequency is stored graphically in the
            # contour attribute of class Syllable, e.g.:
            #   "_/"    (even + rising)
            #   "/_"    (rising + even)
            #   "\\_/"  (lowering, even, rising)
            # So, we have to distribute this contour in all the time provided for
            # the syllable.
            # It may be the case that all contours are even, i.e. the user has not
            # provided any information about intonation. The current implementation
            # handles it in the simplest way: tonality follows the 2-3-1 pattern
            # then.
            # Syllable boundaries can be obtained directly in self.temporal_grid():
            # each boundary is marked by the "| " signed prepended to the name of
            # the phoneme in production (phoneme names are stored in gridlabels).
            # The syllable lasts until the next syllable or a pause. Pauses are
            # marked in griddecipher, as "#".
            instructions = list()
            instructionsgiven = False
            # Preparing to distribute the contours over syllables.
            # Every syllable is divided into the following parts:
            #       Onset       The first group of consonants
            #       Nucleus     The syllable-forming vowel
            #       Coda        The second group of consonants
            # The onset and coda may be absent, but nucleus should be present.
            # The vowel is produced at a constant voice pitch unless this vowel is
            # long.
            # If the vowel is of a regular duration, we have to distribute the
            # contour information between the onset and coda.
            for synt in self.synts:
                for syll in synt.sylls:
                    contour = syll.contour
                    if contour != "_":
                        instructionsgiven = True
                    if len(contour) == 1:
                        onset, nucleus, coda = contour, "_", contour
                        for k, dur in enumerate(syll.lengths):
                            if dur == "Long" and syll.constituents[k].phclass == "V":
                                nucleus += contour
                    elif len(contour) == 2:
                        onset, nucleus, coda = contour[0], "_", contour[1]
                        for k, dur in enumerate(syll.lengths):
                            if dur == "Long" and syll.constituents[k].phclass == "V":
                                nucleus += contour[1]
                    else: # len(contour) == 3 or more; contour is cut at position 3
                        onset, nucleus, coda = contour[0], contour[1], contour[2]
                        if nucleus != "_":
                            for k, dur in enumerate(syll.lengths):
                                if dur == "Long" and syll.constituents[k].phclass == "V":
                                    nucleus = "_" + contour[1]
                            if len(nucleus) == 1: # The vowel is not long
                                nucleus = "_"
                                coda = contour[1] + contour[2]
                    instructions.append((onset, nucleus, coda))
            if instructionsgiven:
                lvl = 133 # lvl = 133 # Hz: the value taken for an average male French speaker
                # from Erwan Pepiot. Male and female speech: a study of mean f0,
                # f0 range, phonation type and speech rate in Parisian French and
                # American English speakers. Speech Prosody 7, May 2014, Dublin,
                # Ireland. pp.305-309, 2014.
                currid = -1
                for t, prevad, ad, nextad in \
                        zip(g, ["#"]+addr, addr, addr[1:]+["#"])[2:]:
                    prevl = lines[-1]
                    prevt = int(prevl[:prevl.find(" ")])
                    if ad != "#":
                        q, n, k = ad
                        currphoneme = self.synts[q].sylls[n].constituents[k]
                    if prevad != "#":
                        q, n, k = prevad
                        prevphoneme = self.synts[q].sylls[n].constituents[k]
                    if nextad != "#":
                        q, n, k = nextad
                        nextphoneme = self.synts[q].sylls[n].constituents[k]
                    if ad == "#": # addresses are "#" only during silence
                        if nextad != "#": # We are ending a silent period
                            lines.append("{} 0 LIN".format(t))
                        elif prevad == "#" and \
                                     not prevl.endswith(" 0 LIN"):
                            # We are inside a silent period but have not marked so
                            # yet
                            lines.append("{} 0 LIN".format(t))
                        elif prevad != "#":
                            if prevphoneme.phclass != "V":
                                # Treating the coda case
                                if len(coda) == 2:
                                    coef = [0, 0]
                                    for num, ch in enumerate(list(coda)):
                                        if ch == "/":
                                            coef[num] = 0.15
                                        elif ch == "\\":
                                            coef[num] = -0.15
                                    lvl += int(coef[0]*0.5*(t-prevt))
                                    lines.append("{} {} LIN".format(int(0.5*(t+prevt)), lvl))
                                    lvl += int(coef[1]*0.5*(t-prevt))
                                    lines.append("{} {} LIN".format(t, lvl))
                                else: # coda contour is a single character
                                    coef = 0
                                    if coda == "/":
                                        coef = 0.15
                                    elif coda == "\\":
                                        coef = -0.15
                                    lvl += int(coef*(t-prevt))
                                    lines.append("{} {} LIN".format(t, lvl))
                            else:
                                # Treating the nucleus case
                                if len(nucleus) == 2: # else: nucleus = "_"
                                    coef = [0, 0]
                                    for num, ch in enumerate(list(nucleus)):
                                        if ch == "/":
                                            coef[num] = 0.1
                                        elif ch == "\\":
                                            coef[num] = -0.1
                                    lvl += int(coef[0]*(t-prevt)*10.0/17)
                                    lines.append("{} {} LIN".format(int((10.0*t+7.0*prevt)/17), lvl))
                                    lvl += int(coef[1]*(t-prevt)*7.0/17)
                                lines.append("{} {} LIN".format(t, lvl))
                    else:
                        # If this is the first phoneme in a syllable,
                        #       if it is a consonant, it begins the onset;
                        #       if it is a vowel, it begins the nucleus.
                        if prevad == "#": 
                            # New syllable after a pause
                            currid += 1
                            onset, nucleus, coda = instructions[currid]
                            if lvl < 100:
                                lvl = 100
                            elif lvl > 230:
                                lvl = 230
                            lines.append("{} {} LIN".format(t, lvl))
                        elif ad[1] != prevad[1]:
                            # New syllable. Treating the previous one
                            if prevphoneme.phclass != "V":
                                # The last syllable ended with coda
                                if len(coda) == 2:
                                    coef = [0, 0]
                                    for num, ch in enumerate(list(coda)):
                                        if ch == "/":
                                            coef[num] = 0.15
                                        elif ch == "\\":
                                            coef[num] = -0.15
                                    lvl += int(coef[0]*0.5*(t-prevt))
                                    if lvl < 100:
                                        lvl = 100
                                    elif lvl > 230:
                                        lvl = 230
                                    lines.append("{} {} LIN".format(int(0.5*(t+prevt)), lvl))
                                    lvl += int(coef[1]*0.5*(t-prevt))
                                else: # coda contour is a single character
                                    coef = 0
                                    if coda == "/":
                                        coef = 0.15
                                    elif coda == "\\":
                                        coef = -0.15
                                    lvl += int(coef*(t-prevt))
                            else:
                                # The last syllable ended on its nucleus.
                                if len(nucleus) == 2: # Else nucleus == 0, coef = 0
                                    coef = [0, 0]
                                    for num, ch in enumerate(list(nucleus)):
                                        if ch == "/":
                                            coef[num] = 0.1
                                        elif ch == "\\":
                                            coef[num] = -0.1
                                    lvl += int(coef[0]*(t-prevt)*10.0/17)
                                    if lvl < 100:
                                        lvl = 100
                                    elif lvl > 230:
                                        lvl = 230
                                    lines.append("{} {} LIN".format(int((10.0*t+7.0*prevt)/17), lvl))
                                    lvl += int(coef[1]*(t-prevt)*7.0/17)
                            if lvl < 100:
                                lvl = 100
                            elif lvl > 230:
                                lvl = 230
                            lines.append("{} {} LIN".format(t, lvl))
                            currid += 1
                            onset, nucleus, coda = instructions[currid]
                        elif prevphoneme.phclass == "C" and \
                                currphoneme.phclass == "V":
                            # Treating the onset case
                            coef = 0
                            if onset == "/":
                                coef = 0.1
                            elif onset == "\\":
                                coef = -0.1
                            lvl += int(coef*(t-prevt))
                            if lvl < 100:
                                lvl = 100
                            elif lvl > 230:
                                lvl = 230
                            lines.append("{} {} LIN".format(t, lvl))
                        elif prevphoneme.phclass == "V" and \
                                currphoneme.phclass == "C":
                            # Treating the nucleus case
                            if len(nucleus) == 2:
                                coef = [0, 0]
                                for num, ch in enumerate(list(nucleus)):
                                    if ch == "/":
                                        coef[num] = 0.1
                                    elif ch == "\\":
                                        coef[num] = -0.1
                                lvl += int(coef[0]*(t-prevt)*10.0/17)
                                lines.append("{} {} LIN".format(int((10.0*t+7.0*prevt)/17), lvl))
                                lvl += int(coef[1]*(t-prevt)*7.0/17)
                                lines.append("{} {} LIN".format(t, lvl))
                            else:
                                coef = 0
                                lines.append("{} {} LIN".format(t, lvl))
            else:
                lines.append("{} 110 LIN".format(g[0]+int(0.6*(g[-1]-g[0]))))   # MANIPULATIONS WITH F0
                lines.append("{} 100 LIN".format(g[-1]))                        # MANIPULATIONS WITH F0
        else:
            slug = self.express(False, False, True, 30, True)
            matfile = os.path.join(copymatfrom, slug + ".mat")
            try:
                uttxfile = os.path.join(copygridfrom, slug + ".uttx")
            except:
                print "Unable to process .mat files with true f0 and lch data without a time grid. Error: copygridfrom == \"OFF\"."
                raise
            with open(uttxfile, "r") as uttx:
                # dataentries = uttx.readline()
                delay = uttx.readline()
                tshift = int(delay[delay.find(" ")+1:])
            observ = imp(matfile, "to", "fo", tshift, 110)      # MANIPULATIONS WITH F0
            join = "COS"
            for moment, coef in observ:
                lines.append("{} {} LIN".format(moment, coef))
        lastl = lines[-1]
        lastt = int(lastl[:lastl.find(" ")])
        lines.append("{} 0 SET".format(lastt+10))
        linesmerged = [lines[0]]
        def cut_command(command):
            return command[command.find(" "):]
        def same_values(command1, command2, command3):
            return cut_command(command1) == cut_command(command2) == cut_command(command3)
        for currl, nextl in zip(lines[1:], lines[2:]):
            lastl = linesmerged[-1]
            if same_values(lastl, currl, nextl):
                continue
            else:
                linesmerged.append(currl)
        linesmerged.append(lines[-1])
        inttext = str(len(lines))+"\n" + "\n".join(lines)
        with open(intonfile, "w") as intf:
            intf.write(inttext)
            intf.close()
        if status:
            status += "There, file \"{}\" ".format(filename)
            status += "regulating the intonation contour has been created.\n"
        else:
            status += "File \"{}\" regulating the intonation ".format(filename)
            status += "contour has been created in the "
            status += "\"{}/\" folder in \"{}/\".\n".format(which, where)
        if vocal:
            print status
        else:
            return status


    def record_phonetic_description(self, phonfile="Syntheses/{}_{}_data/{}_{}.utt",
                       t0=2000, pause=40, iterstep=10, vocal=True,
                       coartmode="COMPLEX", speechrate="Normal",
                       copygridfrom="OFF"):
        """Records a file explaining the temporal grid.
        Input:
            phonfile: string, the path for the output file.
                By default, it is set as "Syntheses/{}_{}_data/{}_{}.utt".
            t0: integer, the moment when utterance production should begin,
                in ms. By default, t0 is 2000 [ms].
            pause: integer, number of milliseconds for a pause between syntagms
                By default, pause is 40 [ms].
            iterstep: integer. A new articulatory vector is formed at least
                every iterstep ms. By default, iterstep is 10 [ms].
            vocal: boolean. Whether to print the status of the function when it
                finishes. By default, yes (True).
            coartmode: string, the mode of the coarticulation model.
                Expected values:
                - "LIN": linear transition between target vectors;
                - "COS": cosine (smoother) transition between target vectors;
                - "COMPLEX": cosine transition with finer operation of
                    articulators.
                By default, it is "COMPLEX".
                Currently it is a dummy argument because there is no difference
                between coarticulation modes in vocal folds operation.
            speechrate: string, a dummy argument for regulating speech rate.
                Currently, all speech is synthesized at a normal rate.
            copygridfrom: string. Regulates whether the control should be
                imported from an external file. By default, no ("OFF");
                otherwise, copygridfrom is the folder containing the control files.
        Output:
            status: string describing what has been done if vocal is True;
                otherwise None.
            Writes an *.utt file that explains the temporal grid.
        """
        if self.status == "ERROR":
            return
        if phonfile.endswith("{}_{}_data/{}_{}.utt"):
            slug = self.express(False, False, True, 30, True)
            phonfile = phonfile.format(slug, coartmode, slug, coartmode)
        foldername = phonfile[:phonfile.rfind("/")]
        filename = phonfile[phonfile.rfind("/")+1:]
        where = foldername[:foldername.rfind("/")]
        which = foldername[foldername.rfind("/")+1:]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            status = "The folder \""+which+"/\" has been created in "
            status += "\""+where+"/\".\n"
        else:
            status = str()
        g, d, v, addr = self.temporal_grid(t0, pause, iterstep,
                                        coartmode, speechrate, copygridfrom)
        lines = list()
        for t, label, a in zip(g, d, addr):
            if a != "#":
                q, n, k = a
                lines.append("{}.   [{}]    {}".format(t,
                                self.synts[q].sylls[n].constituents[k].name,
                                label))
            else:
                lines.append("{}.   #".format(t))
        phtext = str(len(lines)) + "\n" + "\n".join(lines) + "\n"
        with open(phonfile, "w") as phf:
            phf.write(phtext)
            phf.close()
        if status:
            status += "There, file \"{}\" describing every ".format(filename)
            status += "moment in the process of synthesis has been created.\n"
        else:
            status += "File \"{}\" describing every moment ".format(filename)
            status += "in the process of synthesis has been created in the "
            status += "\"{}/\" folder in \"{}/\".\n".format(which, where)
        if vocal:
            print status
        else:
            return status
        
        
    def record_xart_script(self, xartfile=\
                           "Syntheses/generateAFs_{}_{}.xart",
                           uniqartvectfile="Syntheses/{}{}_data/{}_{}.vtpx",
                           t0=2000, pause=40, iterstep=10, vocal=True,
                           coartmode="COMPLEX", speechrate="Normal"):
        """Makes a script for Xarticul to create area function files. 
        Input:
            xartfile: string, the path for the output file.
                By default, it is set as
                "Syntheses/generateAFs_{}_{}.xart".
            artvfile: the path to the file with articulatory vectors.
                By default, it is set as "Syntheses/{}{}_data/{}{}.vtp".
            t0: integer, the moment when utterance production should begin,
                in ms. By default, t0 is 2000 [ms].
            pause: integer, number of milliseconds for a pause between syntagms
                By default, pause is 40 [ms].
            iterstep: integer. A new articulatory vector is formed at least
                every iterstep ms. By default, iterstep is 10 [ms].
            vocal: boolean. Whether to print the status of the function when it
                finishes. By default, yes (True).
            coartmode: string, the mode of the coarticulation model.
                Expected values:
                - "LIN": linear transition between target vectors;
                - "COS": cosine (smoother) transition between target vectors;
                - "COMPLEX": cosine transition with finer operation of
                    articulators.
                By default, it is "COMPLEX".
                Currently it is a dummy argument because there is no difference
                between coarticulation modes in vocal folds operation.
            speechrate: string, a dummy argument for regulating speech rate.
                Currently, all speech is synthesized at a normal rate.
        Output:
            status: string describing what has been done if vocal is True;
                otherwise None.
            Writes an xart script for translating *.vtp-articulatory vector
                files into the area functions.
        """
        if self.status == "ERROR":
            return
        if xartfile.endswith("Syntheses/generateAFs_{}_{}.xart") and \
                uniqartvectfile.endswith("{}{}_data/{}{}.vtpx"):
            slug = self.express(False, False, True, 30, True)
            xartfile = xartfile.format(slug, coartmode)
            uniqartvectfile = uniqartvectfile.format(slug, coartmode, slug, coartmode)
        foldername = xartfile[:xartfile.rfind("/")]
        filename = xartfile[xartfile.rfind("/")+1:]
        where = foldername[:foldername.rfind("/")]
        which = foldername[foldername.rfind("/")+1:]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            status = "The folder \""+which+"/\" has been created in "
            status += "\""+where+"/\".\n"
        else:
            status = str()
        text = "setDisplayScale(2.5);\n"
        modelname = VocalTract().model
        text += "readCurviPCAModel(\"" + modelname[modelname.rfind("/") + 1:] + "\", false, 1.);\n"
        text += "setMRIAntGrid();\n"
        text += "setVelumAsVTWall();\n"
        text += "curviModelSetNbTongueLinComp4Reconstruction(12);\n"
        text += "readSequenceOfArticulatoryVectors(\"{}\");\n".format(uniqartvectfile[uniqartvectfile.rfind("/")+1:])
        text += "generateAreaFunctionsFromArticulatoryVectors"
        wheretoputtheAFs = uniqartvectfile[uniqartvectfile.find("/")+1:uniqartvectfile.rfind("/")]
        if wheretoputtheAFs == "":
            wheretoputtheAFs = uniqartvectfile[uniqartvectfile.rfind("/")+1:uniqartvectfile.rfind(".")] + "_data"
        text += "(\""+wheretoputtheAFs+"\");\n"
        with open(xartfile, "w") as xartf:
            xartf.write(text)
            xartf.close()
        if status:
            status += "There, file \"{}\" containing ".format(filename)
            status += "instructions for Xarticul to create area function files"
            status += " has been created.\n"
        else:
            status += "File \"{}\" containing instructions ".format(filename)
            status += "for Xarticul to create area function files has been "
            status += "created in the "
            status += "\"{}/\" folder.\n".format(which)
        if vocal:
            print status
        else:
            return status


    def record(self, outputloc="Syntheses/", t0=2000, pause=40, iterstep=10,
               vocal=True, coartmode="COMPLEX", speechrate="Normal",
               xarticul=True, visualise=True, calculateAFs=False,
               copymatfrom="OFF", copygridfrom="OFF", totime=None, corpuslogger=str()):
        """Records the constructed utterance according to the chosen
            coarticulation mode coartmode.
        Input:
            outputloc: string, the path to a folder where to put the results.
                In outputloc, a folder called "{utterance}_{mode}_data/"
                is going to be created (if it does not exist).
            t0: integer, the moment when utterance production should begin,
                in ms. By default, t0 is 2000 [ms].
            pause: integer, number of milliseconds for a pause between syntagms
                By default, pause is 40 [ms].
            iterstep: integer. A new articulatory vector is formed at least
                every iterstep ms. By default, iterstep is 10 [ms].
            vocal: boolean. Whether to print the status of the function when it
                finishes. By default, yes (True).
            coartmode: string, the mode of the coarticulation model.
                Expected values:
                - "LIN": linear transition between target vectors;
                - "COS": cosine (smoother) transition between target vectors;
                - "COMPLEX": cosine transition with finer operation of
                    articulators.
                By default, it is "COMPLEX".
            speechrate: string, a dummy argument for regulating speech rate.
                Currently, all speech is synthesized at a normal rate.
            xarticul: boolean, whether the file names should obey Xarticul's
                conventions and whether to create xarticul scripts for the
                utterance. By default, yes (True).
            copymatfrom: string. Regulates whether the lch and f0 control should be imported
                from external files. By default, no ("OFF"); otherwise,
                copymatfrom is the folder containing those control files.
            copygridfrom: string. Regulates whether the grid human-written control should be imported
                from external files. By default, no ("OFF"); otherwise,
                copymatfrom is the folder containing those control files.
        Output:
            None if vocal is True,
            otherwise status: string describing what has been done.
        Inside the utterance-specific folder in outputloc, we create:
            * "{utterance}_{mode}.vtp": list of articulatory vectors
                corresponding to the utterance.
                A single target vector can be obtained as ph.artv[context],
                where ph in syll.constituents, where syll in synt.sylls,
                and context in contexts, and contexts in 
                synt.anticipations, and synt in self.synts.
                These target articulatory vectors are to be manipulated
                according to coartmode, resulting in a sequence of vectors
                to be stored in an vtp-file.
            * "{utterance}_{mode}.af": list of the corresponding area
                function files.
                Currently, the area function files, called AF00...{ID},
                are produced by Xarticul.
            * "{utterance}_{mode}.ag0": operating glottal opening over
                time for producing the utterance.
            * "{utterance}_{mode}.agp": operating vocal folds oscillations
                over time for producing the utterance.
            * "{utterance}_{mode}.f0": operating the intonation contour.
        """
        if totime:
            totime.append(time.time())
        if self.status == "ERROR":
            if vocal:
                print "No synthesis files have been created."
                return
            else:
                return "No synthesis files have been created."
        slug = self.express(False, False, True, 30, True)
        if xarticul:
            slug = slug.replace("-", "")
            slug = slug.replace("(...)", "")
        status = "Utterance " + self.express(False, protectcurlybraces=True) + "...\n\n"
        if totime:
            status += "--- Starting over with processing of the utterance: {} s.\n".format(totime[-1]-totime[-2])
        where = "{}{}{}_data/" if xarticul else "{}{}_{}_data/"
        where = where.format(outputloc, slug, coartmode)
        if not os.path.exists(where):
            os.makedirs(where)
            if xarticul:
                status += "Folder \"{}{}_data/\" has been created in \"{}\".\n"
            else:
                status += "Folder \"{}_{}_data/\" has been created in \"{}\".\n"
            status = status.format(slug, coartmode, outputloc)
        name = where+"{}{}".format(slug, coartmode) if xarticul else \
                where+"{}_{}".format(slug, coartmode)
        artvectfile, uniqartvectfile, affile, glpressfile, vocfoldsfile, \
            intonfile, phonfile = name+".vtp", name+".vtpx", name+".af", \
                                    name+".ag0", name+".agp", name+".f0", name+".utt"
        if xarticul:
            xartfile = outputloc+"generateAFs_{}_{}.xart".format(slug, coartmode)
        if totime:
            totime.append(time.time())
        status += self.record_art_vectors(artvectfile, t0, pause, iterstep,
                                          False, coartmode, speechrate, copygridfrom)
        if totime:
            totime.append(time.time())
            status += "--- Articulatory vector recording: {} s.\n".format(totime[-1]-totime[-2])
        status += self.record_af_list(affile, artvectfile, uniqartvectfile, t0,
                                      pause, iterstep, False, coartmode,
                                      speechrate, copygridfrom, corpuslogger)
        if xarticul and corpuslogger==str():
            copy2(uniqartvectfile, outputloc+"{}{}.vtpx".format(slug, coartmode))
        if totime:
            totime.append(time.time())
            status += "--- Area functions list generation: {} s.\n".format(totime[-1]-totime[-2])
        status += self.record_glottal_opening(glpressfile, t0, pause,
                                                  iterstep, False, coartmode,
                                                  speechrate, copymatfrom, copygridfrom)
        if totime:
            totime.append(time.time())
            status += "--- Glottal pressure management: {} s.\n".format(totime[-1]-totime[-2])
        # The glottal chink model does not use the agp files.
        status += self.record_vocal_folds(vocfoldsfile, t0, pause, iterstep,
                                                 False, coartmode, speechrate,
                                                 copygridfrom)
        if totime:
            totime.append(time.time())
            status += "--- Vocal folds vibrations management: {} s.\n".format(totime[-1]-totime[-2])
        status += self.record_intonation(intonfile, t0, pause, iterstep,
                                                 False, coartmode, speechrate,
                                                 copymatfrom, copygridfrom)
        if totime:
            totime.append(time.time())
            status += "--- Prosody management: {} s.\n".format(totime[-1]-totime[-2])
        status += self.record_phonetic_description(phonfile, t0, pause, iterstep,
                                                 False, coartmode, speechrate,
                                                 copygridfrom)
        if totime:
            totime.append(time.time())
            status += "--- Generation of an accompanying utt-file: {} s.\n".format(totime[-1]-totime[-2])
        if xarticul and corpuslogger==str():
            status += self.record_xart_script(xartfile, uniqartvectfile, t0,
                                              pause, iterstep, False,
                                              coartmode, speechrate)
            if totime:
                totime.append(time.time())
                status += "--- Generation of an xarticul script: {} s.\n".format(totime[-1]-totime[-2])
        status = status.replace("{{", "{").replace("}}", "}")
        if visualise and corpuslogger==str():
            visfolder = "{}{}{}_vis/" if xarticul else "{}{}_{}_vis/"
            visfolder = visfolder.format(outputloc, slug, coartmode)
            statusvis = str()
            if not os.path.exists(visfolder):
                os.makedirs(visfolder)
                if xarticul:
                    statusvis += "Folder \"{}{}_vis/\" has been created in \"{}\".\n"
                else:
                    statusvis += "Folder \"{}_{}_vis/\" has been created in \"{}\".\n"
                statusvis = statusvis.format(slug, coartmode, outputloc)
            visualisationInterface.createimages(artvectfile,                        # scriptFileName:   the vtp file
                                                visfolder,                          # directory:        where to put the images
                                                VocalTract().model,                 # modelFile:        the VT model
                                                True,                               # derivedFromCurrentImages
                                                2,                                  # scale
                                                12,                                 # nbFactors
                                                False,                              # showPic
                                                True,                               # createGIF
                                                True,                               # saveSeparateFrame
                                                calculateAFs,                       # calculateAFs
                                                totime[-1:] if totime else totime,  # totime
                                                corpuslogger)                       # corpuslogger
            if totime:
                totime.append(time.time())
                if calculateAFs:
                    statusvis += "--- Visualising the sequence and calulating the area functions, overall: {} s.\n".format(totime[-1]-totime[-2])
                else:
                    statusvis += "--- Visualising the sequence, overall: {} s.\n".format(totime[-1]-totime[-2])
            status += statusvis + "Visualisation of the utterance {} complete.\n\n".format(self.express(False))
        if vocal:
            print status
        if not corpuslogger:
            return status
        else:
            return status, phonfile, affile


class Corpus(object):
    def __init__(self, name="Corpus", utterances=list(), folderpath="Data/Speech-Synthesis-Database/DB-Full/",
                 extension=".artv"):
        self.utterances = [Utterance(utt, folderpath, extension) for utt in utterances]
        self.corpusname = name
        self.logger = self.corpusname + ".info"
        self.vectors = self.corpusname + ".vtpc"
        self.t0 = None
    

    def append(self, moreutterances, folderpath="Data/Speech-Synthesis-Database/DB-Full/",
                 extension=".artv"):
        self.utterances += [Utterance(phrase, folderpath, extension) for utt in moreutterances]

        
    def record(self, utterances=list(), outputloc="Syntheses/", t0=2000, pause=40, iterstep=10,
               vocal=True, coartmode="COMPLEX", speechrate="Normal",
               xarticul=True, visualise=True, calculateAFs=False,
               copymatfrom="OFF", copygridfrom="OFF", totime=None,
               folderpath="Data/Speech-Synthesis-Database/DB-Full/", extension=".artv"):
        ensure_dir(outputloc)
        self.corpusloc = outputloc
        delete_if_exist([outputloc + self.logger, outputloc + self.vectors]) 
        self.intermediateAFs = outputloc + self.corpusname + "_data/"
        utterancestorecord = [Utterance(phrase, folderpath, extension) for utt in utterances] if utterances else list(self.utterances)
        self.t0 = t0
        self.logger = os.path.join(outputloc, (self.corpusname + ".info").format(coartmode))
        self.vectors = os.path.join(outputloc, (self.corpusname + ".vtpc").format(coartmode))
        self.uttfiles, self.affiles = list(), list()
        for utt in utterancestorecord:
            _, uttfile, affile = utt.record(outputloc, t0, pause, iterstep, vocal, coartmode,
                       speechrate, xarticul, visualise, calculateAFs, copymatfrom, copygridfrom, totime, self.logger)
            self.uttfiles.append(uttfile)
            self.affiles.append(affile)
        # with open(self.vectors, "w+") as v:
        #     content = v.read()
        #     prefix = str(VocalTract().totalparams) + "\n"
        #     if not content.startswith(prefix):
        #         content = prefix + content
        #     v.seek(0)
        #     v.write(content)
        Utterance("a", folderpath, extension).record_xart_script(outputloc+"generateAFs_{}.xart".format(self.corpusname),
                                                                 self.vectors, t0, pause, iterstep, False, coartmode,
                                                                 speechrate)
    
    
    def process_AFs(self, loc=str(), logger=str(), vectors=str()):
        if not (loc or logger or vectors):
            logger = self.logger
            vectors = self.vectors
            loc = self.logger[:self.logger.rfind(".")] + "_data/"
        correct_xa_files(loc, self.t0)
        with open(logger) as l:
            loc2glob = dict()
            directives = list()
            for line in l.readlines():
                if " --> " in line:
                    src, dst = line.strip().split(" --> ")
                    loc2glob[dst] = src
                    directives.append([src, dst])
        with open(loc + "afcorrection.log", "w") as logf:
            pass
        for uttfile, affile in zip(self.uttfiles, self.affiles):
            correct_xax_files(loc, loc, uttfile, affile, self.t0, loc2glob)
        # newlog += "{}AF{}.xax --> {}AF{}.xax\n".format(corpusdir, format(corpindexer[k], 5), localdir, format(afindexer[k], 5))
        for src, dst in directives:
            copy2(src, dst)
        produce_matlab_list(self.corpusloc)


# def connect_to_existing_corpus("newcorpus")


def fetch_phoneme_by_name(phonemelist, name):
    """From a given list of Phonemes, fetches the one (the first one) with the
    given name.
    Input: phonemelist: list of Phoneme, name: string.
    Output: Phoneme.
    """
    for ph in phonemelist:
        if ph.name == name:
            return ph
    print "Phoneme [{}] has not been found in the provided list of objects.".format(name)


def scan_directory_for_phonemes(folderpath=
                                "Data/Speech-Synthesis-Database/DB-Full/",
                                extension=".artv", sort=True):
    """Scans for *.extension files in folderpath and produces a list of Phoneme
        in the required form that is determined by sort.
    Input:
        folderpath:string, extension:string - the location of the files to
            look for and their expected type.
            They can be set to "Data/Speech-Synthesis-Database/DB-Full/" and
            ".artv".
        sort: boolean: whether to sort them into consonants, vowels and silence
            By default, yes (True).
    Output: list of Phoneme if sort is False, three-element tuple of two lists
        of Phoneme and one Phoneme (or None in place of that) if sort is True.
    """
    dircontents = os.listdir(folderpath)
    if not sort:
        phs = list()
        for el in dircontents:
            if el.endswith(extension): # Select only the necessary files.
                el = el[:el.rfind(".")]
                phs.append(Phoneme(el, folderpath, extension))     
        return phs
    consonants = list()
    vowels = list()
    silence = None
    for el in dircontents:
        if el.endswith(extension): # Select only the necessary files.
            el = el[:el.rfind(".")]
            phoneme = Phoneme(el, folderpath, extension)
            if phoneme.phclass == "C":
                consonants.append(phoneme)
            elif phoneme.phclass == "V":
                vowels.append(phoneme)
            else:
                silence = phoneme
    return (consonants, vowels, silence)


def slice(phonemes, keys, articulators):
    """Slices a set of articulatory vectors, returning only those parameters
        that are related to a particular articulator or a particular set of
        articulators.
    Input:
        phonemes: list of Phoneme.
        keys: list of names that the elements from "phonemes" anticipate.
            phoneme.artv[key] will result in an articulatory vector that
            the program will deal with.
        articulators: string or list of strings. Expected arguments: see the
            description of the constructor in the VocalTract class.
    Output: vector of float.
    """
    vt = VocalTract()
    etalone = vt.articulators
    if articulators != list(articulators):
        articulators = [articulators]
    for art in articulators:
        if not etalone.has_key(art):
            print "The articulator \'"+art+"\' in sequence ["+\
                    "-".join([ph.name for ph in phonemes])+\
                    "] is not recognized."
            return
    articulators = vt.reorder_articulators(articulators)
    return [ph.slice(articulators, k) for ph, k in zip(phonemes, keys)]


def fetch_semivowels(phonemes):
    """Gets semivowels from a list of phonemes.
    Input: list of Phoneme.
    Output: list of Phoneme.
    """
    return [ph for ph in phonemes if ph.artmanner == "Semivowel"]


def database_expand(inputfolderpath=
                    "Data/Speech-Synthesis-Database/DB-Before-Expansion/",
                    outputfolderpath="Data/Speech-Synthesis-Database/DB-Full/",
                    inputextension=".dat", outputextension=".artv"):
    """Adds extra samples to the database, estimating them from the corner
        vowels.
    Input:
        inputfolderpath:string is the name of the folder with the vectors.
            By default, "Data/Speech-Synthesis-Database/DB-Before-Expansion/".
        outputfolderpath:string is the name of the folder where the system
            has to put the expanded version of the database. This folder does
            not have to exist before the start of the program.
            By default, "Data/Speech-Synthesis-Database/DB-Full/".
        inputextension:string is used as the identifier of which files to use.
            See that this ending does not coincide with the ones of the of the
            other files that you may keep in the same folder.
            By default, extension is set as ".dat"
        outputextension:string is the extension for the output files containing
            the expanded database entries.
            By default, extension is set as ".artv" Such a file may be opened
            in a text editor.
    Output:
        None
        Creates outputfolderpath if it does not exist.
        Writes an expanded version of the database there, with the file
            extension outputextension.
    """
    if not os.path.exists(outputfolderpath):
        os.makedirs(outputfolderpath)
    consonants, vowels, silence = \
                scan_directory_for_phonemes(inputfolderpath, inputextension)
    for semivowel in fetch_semivowels(consonants):
        consonants.remove(semivowel)
        vowels.append(semivowel)
    vowelnames = [ph.name for ph in vowels]
    for consonant in consonants:
        copy2(inputfolderpath + consonant.name + inputextension,
                outputfolderpath + consonant.name + outputextension)
        anticipatedvowels = consonant.artv.keys()
        if "u" in anticipatedvowels \
                and "i" in anticipatedvowels \
                and "a" in anticipatedvowels:
            anticipcorner = [consonant.artv["u"],
                             consonant.artv["i"],
                             consonant.artv["a"]]
            toestimate = get_missing_elements(anticipatedvowels, vowelnames)
            if toestimate:
                estimations = "\n"
                for item in toestimate:
                    absentvowel = fetch_phoneme_by_name(vowels, item)
                    coeffs = absentvowel.proj
                    estim = estimate_from_corner_vowels(anticipcorner, coeffs)
                    estim = " ".join([str(el) for el in estim])
                    estimations += "{} + {} : ".format(consonant.name, item)
                    estimations += "{}".format(estim)
                    estimations += " // estimated from anticipation of corner vowels.\n"
                with open(outputfolderpath + consonant.name + outputextension,
                          "a") as addition:
                    addition.write(estimations)
    for vowel in vowels:
        copy2(inputfolderpath + vowel.name + inputextension,
            outputfolderpath + vowel.name + outputextension)
    if silence:
        copy2(inputfolderpath + silence.name + inputextension,
            outputfolderpath + silence.name + outputextension)


def compare_articulatory_vectors(real, estimated):
    """Compares two articulatory vectors. A function that is used to evaluate
    the similitude of the real sample with an estimated one.
    Input:
        real, estimated: lists of float
    Output:
        totalerr: float
        articulerr: list of floats that are the contribution of a particular
            articulator to the difference between the vectors,
            in agreement with the articulators as defined by class VocalTract,
            constructed with the alphabetical order, over the full articulator
            names.
    """
    vt = VocalTract()
    parameters = vt.fullnarticulators # {name of artic: [par#, par#, ...]}
    articulators = vt.reorder_articulators(parameters.keys(), False)
    totalerr = euclid_dist(real, estimated)
    if totalerr <= 0.00001:
        return 0.0,  [0.0 for art in articulators]
    articulerr = list()
    for art in articulators:
        realpart = [real[k] for k in parameters[art]]
        estimatedpart = [estimated[k] for k in parameters[art]]
        x = euclid_dist(realpart, estimatedpart)
        contribution = 100*x*x/(totalerr*totalerr)
        contribution = float("{0:.2f}".format(contribution))
        articulerr.append(contribution)
    return totalerr, articulerr
    

def evaluate_projections(inputfolderpath=
                    "Data/Speech-Synthesis-Database/DB-Before-Expansion/",
                    outputfolderpath=
                    "Data/Projections/Evaluation-Of-Corner-Vowels-Assumption/Visualisations & Reports/",
                    inputextension=".dat", outputextension=".evpr"):
    """Compares, when possible, the estimated samples from the ones that
    picture anticipation of corner vowels to the real ones.
    Input:
        inputfolderpath:string is the name of the folder with the vectors.
            By default, "Data/Speech-Synthesis-Database/DB-Before-Expansion/".
        outputfolderpath:string is the name of the folder where the system
            has to put the estimations.
            By default, "Data/Projections/Evaluation-Of-Corner-Vowels- \
                Assumption/Visualisations & Reports/".
        inputextension:string is used as the identifier of which files to use.
            See that this ending does not coincide with the ones of the of the
            other files that you may keep in the same folder.
            By default, extension is set as ".dat"
        outputextension:string is the extension for the output files containing
            evaluation of the corner vowel approach.
            By default, extension is set as ".evpr". Such a file may be opened
            in a text editor.
    Output:
        None
        Creates outputfolderpath if it does not exist.
        Writes the evaluation files there, with the given file extension.
    """
    if not os.path.exists(outputfolderpath):
        os.makedirs(outputfolderpath)
    consonants, vowels, silence = \
                scan_directory_for_phonemes(inputfolderpath, inputextension)
    for semivowel in fetch_semivowels(consonants):
        consonants.remove(semivowel)
        vowels.append(semivowel)
    vowelnames = [ph.name for ph in vowels]
    vt = VocalTract()
    articulators = vt.reorder_articulators(vt.fullnarticulators.keys(), False)
    for consonant in consonants:
        anticipatedvowels = consonant.artv.keys()
        if "u" in anticipatedvowels \
                and "i" in anticipatedvowels \
                and "a" in anticipatedvowels:
            anticipcorner = [consonant.artv["u"],
                             consonant.artv["i"],
                             consonant.artv["a"]]
            toestimate = [vow for vow in anticipatedvowels if vow != "Solo"]
            if toestimate:
                consfolder = outputfolderpath + consonant.name + "/"
                if not os.path.exists(consfolder):
                    os.makedirs(consfolder)
                estimations = "\n"
                for item in toestimate:
                    estimatedvowel = fetch_phoneme_by_name(vowels, item)
                    coeffs = estimatedvowel.proj
                    estimation = estimate_from_corner_vowels(anticipcorner, coeffs)
                    estimations += "Quality of the estimation for "
                    estimations += "{} + {}:\n".format(consonant.name, item)
                    totalerr, articulerr = compare_articulatory_vectors(
                                            consonant.artv[item], estimation)
                    if totalerr <= 0.00001:
                        estimations = estimations[:-1]
                        estimations += " 100%.\n\n"
                        continue
                    if not os.path.exists(consfolder+"output-"+consonant.name+
                                          "-"+item+"/"):
                        os.makedirs(consfolder+"output-"+consonant.name+"-"+
                                    item+"/")
                    with open(consfolder+"output-"+consonant.name+"-"+item+"/"+consonant.name+"-"+item+".vtp", "w") as vtp:
                        vtpdata = "26\n" # Warning: not a parameter!
                        vtpdata += " ".join([str(el) for el in consonant.artv[item]])
                        vtpdata += "\n"
                        vtpdata += " ".join([str(el) for el in estimation])
                        vtpdata += "\n"
                        vtp.write(vtpdata)
                    estimations += "    Error:\n"
                    estimations += "    " + str(totalerr) + "\n"
                    for k in range(len(articulators)):
                        art = articulators[k]
                        if art not in vt.subarticulators:
                            estimations += "          - {}:".format(art)
                            estimations += " "*(17-len(art))
                        else:
                            estimations += "             - {}:".format(art)
                            estimations += " "*(14-len(art))
                        estimations += "{}%\n".format(articulerr[k])
                with open(inputfolderpath + consonant.name + inputextension,
                          "r") as contents:
                    lines = contents.readlines()
                    phonemeheader = lines[0] + lines[1] + lines[2]
                with open(consfolder + consonant.name + outputextension,
                          "w") as conclusion:
                    conclusion.write(phonemeheader + estimations)
    vowelu = fetch_phoneme_by_name(vowels, "u").artv["Solo"]
    voweli = fetch_phoneme_by_name(vowels, "i").artv["Solo"]
    vowela = fetch_phoneme_by_name(vowels, "a").artv["Solo"]
    cornervowels = [vowelu, voweli, vowela]
    for vowel in vowels:
        text = "\n"
        coeffs = vowel.proj
        estimation = estimate_from_corner_vowels(cornervowels, coeffs)
        text += "Quality of the estimation:\n"
        totalerr, articulerr = compare_articulatory_vectors(
                                            vowel.artv["Solo"], estimation)
        if totalerr <= 0.00001:
            text += " 100%.\n"
        else:
            if not os.path.exists(outputfolderpath+"/"+vowel.name+"/"):
                os.makedirs(outputfolderpath + "/" + vowel.name + "/")
            with open(outputfolderpath+vowel.name+"/"+
                      vowel.name+".vtp", "w") as vtp:
                vtpdata = "26\n" # Warning: not a parameter!
                vtpdata += " ".join([str(el) for el in vowel.artv["Solo"]])
                vtpdata += "\n"
                vtpdata += " ".join([str(el) for el in estimation])
                vtpdata += "\n"
                vtp.write(vtpdata)
            text += "    Error:\n"
            text += "    " + str(totalerr) + "\n"
            for k in range(len(articulators)):
                art = articulators[k]
                if art not in vt.subarticulators:
                    text += "          - {}:".format(art)
                    text += " "*(17-len(art))
                else:
                    text += "             - {}:".format(art)
                    text += " "*(14-len(art))
                text += "{}%\n".format(articulerr[k])
            with open(inputfolderpath + vowel.name + inputextension,
                              "r") as contents:
                lines = contents.readlines()
                phonemeheader = lines[0] + lines[1] + lines[2]
            with open(outputfolderpath + vowel.name + "/" + vowel.name +
                      outputextension, "w") as conclusion:
                conclusion.write(phonemeheader + text)
