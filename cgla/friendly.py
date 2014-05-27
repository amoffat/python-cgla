# -*- coding: utf8 -*-

from math import pi, acos, asin, atan
from fractions import Fraction




_common_pi_slices = (
    1.0/6,
    1.0/4
)




def match(num, threshold):
    """ takes a number and a 'snapping' threshold and returns a friendlier
    version of that number, if one exists, or None if none exists """
    
    found = whole_num(num, threshold)
    if found:
        return found
    
    found = pi_multiple(num, _common_pi_slices, threshold)
    if found:
        return found
    
    found = trig(num, _common_pi_slices, threshold)
    if found:
        return found
    
    return None




def _approx_equal(v1, v2, threshold):
    return abs(v1 - v2) < threshold


def _whole_num(num, threshold):
    rounded = round(num)
    if _approx_equal(num, rounded, threshold):
        return int(rounded)
    else:
        return None
    
def whole_num(num, threshold):
    found = _whole_num(num, threshold)
    if found is not None:
        return str(found)
    else:
        return None
    
    
def pi_multiple(num, slices, threshold):
    abs_num = abs(num)
    sign = "-" if num < 0 else ""
    
    pi_slices = [(s*pi, s) for s in slices]
    
    for pi_slice, slice in pi_slices:
        divided = abs_num / pi_slice
        numer = _whole_num(divided, threshold)
        if numer:
            denom = int(1.0/slice)
            
            fract = str(Fraction(numer, denom))
            if fract == "1":
                return sign+"π"
            elif "/" in fract:
                numer, denom = fract.split("/")
                if numer == "1":
                    numer = ""
                return sign+"%sπ/%s" % (numer, denom)
            else:
                return sign+"%sπ" % fract
                   
    return None



_trig_mapping = {
    asin: "sin",
    acos: "cos",
    atan: "tan",
}


def _call_trig(fn, num):
    out = None
    try:
        out = fn(num)
    # potential math domain error
    except ValueError:
        pass
    return out


def _smallest_sin_or_cos(num, slices, threshold):
    """ this ugly little function is designed to give us the best
    representation of an angle.  because sine and cosine can both be represented
    in terms of eachother, the first trig fn to run will be the one used to
    represent the an angle.  instead, we want to represent the trig function
    that uses the smallest angle input """
    
    smallest = None
    smallest_friendly = None
    smallest_fn = None
    
    # try cosine
    inverse = _call_trig(acos, num)
    if inverse is not None:
        found = pi_multiple(inverse, slices, threshold)
        if found:
            smallest_friendly = found
            smallest = inverse
            smallest_fn = _trig_mapping[acos]
        
    # now try sin
    inverse = _call_trig(asin, num)
    if inverse is not None:
        found = pi_multiple(inverse, slices, threshold)
        if found and inverse < smallest:
            smallest_friendly = found
            smallest = inverse
            smallest_fn = _trig_mapping[asin]
            
    return smallest_friendly, smallest_fn


def trig(num, slices, threshold):
    """ finds a trig-friendly version of num by looking at the trig function
    that could have been applied to some multiple/fraction of pi """
    
    found, name = _smallest_sin_or_cos(num, slices, threshold)

    if not found:
        inverse = _call_trig(atan, num)
        if inverse is not None:
            found = pi_multiple(inverse, slices, threshold)
            name = _trig_mapping[atan]
        
    if found:
        return "%s(%s)" % (name, found)
        
    return None