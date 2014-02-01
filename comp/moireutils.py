from bpf4 import bpf
from em.lib import returns_tuple
from peach import *
import math
from em import comp


fdn_feedback = bpf.linear(0,0,	10,0.91,	20,0.91,	30,0.91,	40,0.98,	     50,0.98,	55,0.92, 110,0.92)
fdn_feedbackwet = bpf.linear(0,0, 20,0.34,	30,0.8,		40,0.8,	45,0.34, 50,0.34,	60,0, 80,0,		90,0.34,	100,0,		110,0)
fdn_phasewet = bpf.linear(0,0, 45,0,  	 50,0.34,	60,0.7, 70,0, 80,0, 90,0.34, 100,0.34, 110,0.34)
fdn_ringmod	 = bpf.linear(0,0, 70,0, 80,0.8, 90,0, 100,0, 110,0)
	
@returns_tuple("fb fb_wet phase_wet ringmod")
def fdn(fdn):
	"""
	fdn: 0-10
	"""
	n = fdn*10
	fb = fdn_feedback(n)
	fbwet = fdn_feedbackwet(n)
	phasewet = fdn_phasewet(n)
	ringmod = fdn_ringmod(n)
	return fb, fbwet, phasewet, ringmod

def _getrelvalue(bpf, relvalue):
	v1 = bpf.maxvalue()
	v0 = bpf.minvalue()
	v = v0 + relvalue*(v1-v0)
	return v

def knobpos_phase(phase=None, rel=None):
	if rel is not None:
		phase = _getrelvalue(fdn_phasewet, rel)
	z = (fdn_phasewet - phase).zeros()
	if len(z):
		return z/10.

def midikeyb_microtone(resulting_note):
	"""
	resulting_note: the note that should sound, as string or midinote
	"""
	if isinstance(resulting_note, str):
		n = n2m(resulting_note)
	else:
		n = resulting_note
	micro = n - int(n)
	if micro == 0:
		return m2n(n)
	cents = int(math.modf(n)[0] * 100 / 2. + 0.5) * 2
	cent_notes = []
	
	# cents lower than 10
	low_cents = int(math.modf(cents/10.)[0]*10)
	if low_cents:
		note = {2:'C#1', 4:'D1', 6:'D#1', 8:'E1'}[low_cents]
		cent_notes.append(note)
	
	# rest
	binary_notes = ['A0', 'Bb0', 'B0', 'C1']
	hi_cents = cents - low_cents
	b = binrepr(hi_cents)
	for i, char in enumerate(reversed(b)):
		if char == '1':
			cent_notes.append(binary_notes[i])
	return cent_notes

def binrepr(num):
	return str(bin(num))[2:]

def melodica_findsource(difftone, intervals=(2, 3, 4, 5)):
	"""
	find the source of a difference tone in the melodica (37 keys)
	"""
	return comp.difftones_sources(difftone, mindistance=0.5, intervals=intervals, maxnote="F6")
