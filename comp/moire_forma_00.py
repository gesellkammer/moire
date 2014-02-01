#!/usr/bin/env python
from __future__ import division

# local
from collections import namedtuple 
from em.distribute import partition_fib, partition_fib_maximize_partitions
from em.iterlib import izip, pairwise
from bpf4 import bpf, as_bpf, warped
from em import permutations
import numpy
from em.collections import ClassDict
from em.elib import accum, flatten, add_suffix, namedtuple_addcolumn
from em.syntax import *
from operator import isNumberType, add
from em import csvtools
from interpoltools import *
import ordereddict

def Track(**kws): return ClassDict(**kws)
Frame = namedtuple('Frame', 'start dur id track')
Measure = namedtuple('Measure', 'start dur id timesig tempo')
INF = float('inf')


duracion_total = 7 * 60 # segundos
ratios = sorted(partition_fib(n=1, numpartitions=3, homogeneity=0), reverse=True)
durA, durB, durCD = [ratio * duracion_total for ratio in ratios]
durC = int(durCD * 0.6 + 0.5)
durD = int(durCD * 0.49)


######################################################################################################################

with block('<<< FORM >>>'):
    ifib = interpol_fib
    comienzo    = 0
    final       = 1
    climax      = interpol_fib(0.5, 1, final,  0, comienzo)
    desarrollo  = ifib1(0.5, comienzo, climax)
    exposicion_dur = desarrollo - comienzo
    desarrollo_dur = climax - desarrollo
    climax_dur  = interpol_fib(0.5, 1, desarrollo_dur, 0, 0)
    climax_release_dur = climax_dur * 0.1
    climax_release = climax + climax_dur - climax_release_dur
    pedal       = climax + climax_dur
    pedal_dur   = interpol_fib(0.5, 1, climax_dur, 0, 0)
    coda_dur    = interpol_fib(0.5, 0, 0, 1, climax_dur - climax_release_dur)
    reprise     = pedal + pedal_dur
    reprise_dur = interpol_fib(0.5, 0, 0, 1, exposicion_dur)
    desarrollo2 = reprise + reprise_dur
    desarrollo2_dur = desarrollo_dur * (reprise_dur / exposicion_dur)
    coda        = desarrollo2 + desarrollo2_dur
    final       = coda + coda_dur
    form0 = (
        ('comienzo',    comienzo),
        ('desarrollo',  desarrollo),
        ('climax',      climax),
        ('climax_release', climax_release),
        ('pedal', pedal),
        ('reprise', reprise),
        ('desarrollo2', desarrollo2),
        ('coda', coda),
        ('final', final)
    )
    warp_time = bpf.multi(comienzo,0, (pedal+climax)*0.5, 0.68*final, 'linear', final,final)
    scale_ratio = 1 / final
    form = ordereddict.OrderedDict()
    for name, offset in form0:
        form[name] = warp_time(offset) * scale_ratio
    form2 = F = ClassDict('Form', form)
    DUR_CLIMAX = 1.5

with block('<<< TRACKS >>>'):
    # -------------  A  --------------
    A = Track(id='A',
        total_dur=durA,
        maxdur=3, mindur=0.5,
        hom_curve = bpf.halfcos(
            (0, 0),
            (1, 0.2)),
        section_entropy_curve = bpf.halfcos(0, 0.4, 1, 0, exp=0.75),
        sections_order   = '>',
        sections_entropy = 0.,
        num_sections     = 13,
        section_mindur   = lambda T: T.mindur * T.num_sections,
        section_maxdur   = min(durC, durD, durB),
        track_hom        = 0.3,
        frames_orders    = bpf.nointerpol(0, -1, F.reprise, 1),
        compression_curve = bpf.halfcos(
          (0,            1),
          (F.desarrollo, 1),
          (F.climax,     1),
          (F.reprise,    1),
          (F.coda,       1),
          (F.final,      1)
        ),
        normalized_pressence = warped(bpf.halfcos({
          F.comienzo:   1,
          F.desarrollo: 1,
          F.climax:     1,
          F.climax_release: 1,
          F.pedal:      0,
          F.reprise-0.0001:0,
          F.reprise:    1,
          F.coda:       1,
          interpol_fib(0.5, 0, F.coda, 1, F.final): 1,
          F.final:0
        }, exp=1.618
        ))
    )
    A.duration = bpf.multi(
        ( F.comienzo,           ifib1(0.5,
                                      A.mindur,
                                      A.maxdur)             ),
        ( ifib1(-0.5,
                F.comienzo,
                F.desarrollo),  None,       'linear'        ),
        ( F.desarrollo,         A.maxdur,   'expon(1)'    ),
        ( F.climax,             DUR_CLIMAX, 'halfcos(1)'  ),
        ( ifib1(0.5,
                F.climax,
                F.pedal),       None                        ),
        ( F.pedal,              A.mindur*0.5,          'halfcos'       ),
        ( F.reprise-0.0001,     A.mindur*0.5                          ),
        ( F.reprise,            A.mindur,   'halfcos(1)'    ),
        # ( F.desarrollo2,        A.mindur                    ),
        ( F.final,              ifib1(0.5,
                                      A.mindur,
                                      A.maxdur), 'halfcos'  )
        # ( F.final,              None                        )
    )
    A.duration_spread = 1 + bpf.multi(
        ( F.comienzo,  .8 ),
        ( F.desarrollo,.8, 'linear' ),
        ( F.climax, .2, 'halfcos'),
        ( F.reprise - 0.0001, .4 ),
        ( F.reprise, .8 ),
        ( F.coda,    .2 )
    )
    # -----------------  B  --------------------------
    B = Track(id='B',
        total_dur=durB,
        maxdur=3, mindur=0.5,
        hom_curve = bpf.halfcos(
            (0, 0.5),
            (1, 0)
        ),
        section_entropy_curve = bpf.halfcos(0, 0.4, 1, 0, exp=0.75),
        sections_order   = '<',
        sections_entropy = 0.,
        num_sections     = 13,
        section_mindur   = lambda T: T.mindur * T.num_sections,
        section_maxdur   = min(durD, durB),
        track_hom        = 0.2,
        frames_orders    = '<', # bpf.nointerpol(0, -1, F.reprise, 1),
        compression_curve = bpf.halfcos(
          (0,            1),
          (F.desarrollo, 1),
          (F.climax,     1),
          (F.reprise,    1),
          (F.coda,       1),
          (F.final,      1)
        ),
        normalized_pressence = warped(bpf.halfcos({
          F.comienzo:   0,
          F.desarrollo: 0,
          F.climax:     1,
          F.climax_release: 1,
          F.pedal:      1,
          F.reprise-0.0001:1,
          F.reprise:    0,
          F.coda:       0,
          interpol_fib(0.5, 0, F.coda, 1, F.final): 0,
          F.final:0
        }, exp=1
        ))
    )
    T = B
    T.duration = bpf.multi(
        ( F.comienzo,       T.mindur                ),
        ( F.desarrollo,     T.mindur,   'nointerpol'),
        ( F.climax,         DUR_CLIMAX, 'halfcos(2)'),
        ( ifib1(-0.5,
                F.climax,
                F.pedal),   None,       'linear'    ),
        ( F.pedal,          T.maxdur,   'halfcos'   ),
        ( F.reprise-0.001,  None,       'linear'    ),
        ( F.reprise,        T.mindur,   'linear'    ),
        ( F.final,          None )
    )
    T.duration_spread = 1 + bpf.multi(
        ( F.comienzo,  .8 ),
        ( F.coda, 0.2 , 'halfcos(0.8')
    )
    # -------------  C  -------------------
    C = Track(id='C',           # col legno
        total_dur=durC,
        maxdur=2, mindur=0.5,
        hom_curve = bpf.halfcos(
            (0, 0.5),
            (1, 0)
        ),
        section_entropy_curve = bpf.halfcos(0, 0.4, 1, 0, exp=0.75),
        sections_order='<',
        sections_entropy = 0.,
        section_mindur = 4,
        section_maxdur = durC * (1/3),
        track_hom      = 0.0,
        num_sections = 5,
        frames_orders = "<",
        compression_curve = bpf.halfcos(
          (0,            1),
          (F.desarrollo, 1),
          (F.climax,     1),
          (F.reprise,    1),
          (F.coda,       1),
          (F.final,      1)
        ),
        normalized_pressence = warped(bpf.halfcos({
          F.comienzo:   0,
          F.desarrollo: 0,
          F.climax:     0,
          F.climax_release: 0,
          F.pedal:      1/3,
          F.reprise-0.0001:1/3,
          F.reprise:    0,
          interpol_fib(0.3333, 0, F.desarrollo2, 1, F.coda):0,
          F.coda:       1,
          F.final:      1,
        }, exp=1
        ))
    )
    T = C
    T.duration = bpf.multi(
        ( F.comienzo, T.mindur ),
        ( F.coda,     T.maxdur )
    )
    T.duration_spread = 1 + bpf.halfcos(
        ( F.comienzo,   .4 ),
        ( F.coda,       .4 )
    )
    # ---------- D -----------------
    D = Track(id='D',
        total_dur=durC * 0.6,
        maxdur=4, mindur=0.5,
        hom_curve = bpf.halfcos(
            (0, 0.5),
            (1, 0.0)
        ),
        section_entropy_curve = bpf.halfcos(0, 0.4, 1, 0, exp=0.75),
        num_sections    = 2,
        sections_order  = '<',
        track_hom       = 0.0,
        sections_entropy=0.,
        section_mindur = 4 * 0.5,
        section_maxdur = durD,
        frames_orders = "<",
        compression_curve = bpf.halfcos(
          (0,            1),
          (F.desarrollo, 1),
          (F.climax,     1),
          (F.reprise,    1),
          (F.coda,       1),
          (F.final,      1)
        ),
        normalized_pressence = warped(bpf.halfcos({
          F.comienzo:   0,
          F.desarrollo: 0,
          F.desarrollo2:0,
          interpol_fib(0.6666, 0, F.desarrollo2, 1, coda):0,
          F.coda:       1,
          F.final:      1
        }, exp=1
        ))
    )
    T = D
    T.duration = bpf.halfcos(
        ( F.comienzo,    T.mindur ),
        ( F.desarrollo2, T.maxdur)
    )
    T.duration_spread = 1 + bpf.halfcos(
        ( F.comienzo,   .4 ),
        ( F.coda,       .4 )
    )
transformations = [
    ('ABABA',   'ABAAB'),
    ('ACACA',   'ACAAC'),
    ('BABA',    'BAAB'),
    ('CACA',    'CAAC'),
    ('AABAABAA',  'AAABAABA'),
    ('AACAAC',  'ACAAAC'),
    ('AABAAB',  'AAABAB'),
    ('ABAB',    'BAAB'),
    ('BCBC',    'BCCB'),
    ('CCCB',    'CCBC'),
    ('ABCC',    'CABC'),
    ('CCCCA',   'CCCAC'),
    ('CCACCBC',  'CCACBCC'),
    ('BBBBA',   'BBBAB'),
    ('CDCD',    'CDDC'),
    ('DDDC',    'DDCD'),
    ('BCDD',    'DBCD'),
    ('DDDDB',   'DDDBD'),
    ('DDBDDCD',  'DDBDCDD'),
    ('CCCCB',   'CCCBC')
]

def inittrack(track):
    # track.num_sections = len(track.frames_entropies)
    track.section_entropies = track.section_entropy_curve.map(track.num_sections)
    track.minval = track.duration
    track.maxval = track.duration * track.duration_spread
    #track.minval = as_bpf(track.minval)
    #track.maxval = as_bpf(track.maxval)
    # this should come after the sections are created, so that we know
    # how many sections we have
    if isinstance(track.frames_orders, basestring):
        track.frames_orders = [track.frames_orders] * track.num_sections
    elif callable(track.frames_orders):
        track.frames_orders_shape = track.frames_orders
        convert = {1:'<', -1:'>'}.get
        xs = numpy.linspace(0, 1, track.num_sections)
        track.frames_orders = map(track.frames_orders.apply(convert), xs)
def create_sections2(totaldur, numsections, mindur, entropy=0, reverse=False):
    secciones = partition_fib(totaldur, numsections, minval=mindur)
    secciones = permutations.permutation_further_than(secciones, entropy, rand=True)
    if reverse:
        secciones = list(reversed(secciones))
    return secciones
def create_sections(track):
    order = track.sections_order in ('descending', '>')
    maxval_maximum = bpf3.maximum(track.maxval)
    sections_dur = partition_fib(
        n=track.total_dur,
        numpartitions=track.num_sections,
        homogeneity=track.track_hom,
        minval=track.section_mindur,
        maxval=track.section_maxdur)
    if sections_dur is None:
        1 / 0
    assert all(track.section_mindur <= section_dur <= track.section_maxdur for section_dur in sections_dur)
    if track.sections_entropy > 0:
        sections_dur = permutations.unsort(sections_dur, track.sections_entropy)
    if track.sections_order in ('descending', '>'):
        sections_dur = list(reversed(sections_dur))
    return sections_dur
def create_hom(curve, sections):
    xs = numpy.linspace(0, 1, len(sections))
    return as_bpf(curve).map(xs)
def sort_frames(frames, order):
    if isinstance(order, tuple):
        order, ratio = order
        return distribute.mirror_sort(order, ratio)
    else:
        if order == '<':
            return sorted(frames)
        elif order == '>':
            return sorted(frames, reverse=True)
def generate_frames(track):
    section_starts = numpy.linspace(0, 1, track.num_sections)
    track.minvals = track.minval.map(section_starts)
    track.maxvals = track.maxval.map(section_starts)
    # instead of sampling the curves at the beginning of each section, get the average
    section_bounds = numpy.linspace(0, 1, track.num_sections + 1)
    minvals = []
    maxvals = []
    for x0, x1 in pairwise(section_bounds):
        minval = track.minval.integrate_between(x0, x1) / (x1 - x0)
        maxval = track.maxval.integrate_between(x0, x1) / (x1 - x0)
        minvals.append(minval)
        maxvals.append(maxval)
    track.minvals = minvals
    track.maxvals = maxvals
    #frames_list = [partition_fib_maximize_partitions(section, max_homogen, track.dur_range[0]) for section, max_homogen in zip(track.sections_dur, track.sections_hom)]
    frames_list = []
    for section_dur, hom, minval, maxval in zip(track.sections_dur, track.sections_hom, track.minvals, track.maxvals):
        print ">>>", section_dur, hom, minval, maxval
        frames = partition_fib_maximize_partitions(section_dur, hom, minval, maxval, debug=True)
        if frames is not None:
            frames_list.append(frames)
        else:
            print """
            It is not possible to partition the desired section with the given values
            """
            minp, maxp = partition_fib_max_part_estimate_partitions_range(section_dur, hom, minval, maxval)
            1/0
    new_frames_list = []
    for frames, section_entropy, frame_order in zip(frames_list, track.section_entropies, track.frames_orders):
        sorted_frames = sort_frames(frames, frame_order)
        if section_entropy > 0:
            frames = permutations.unsort(sorted_frames, section_entropy)
        new_frames_list.append(frames)
    return new_frames_list

def plot_track(track):
    xs = accum(x for x in flatten(track.frames_list)) |to| list
    ys = flatten(track.frames_list)
    import pylab
    pylab.plot(xs, ys, '.')
    pylab.show()

def apply_transformations(frames, transformations):
    def apply_transform(frames, start, orig, trans):
        from collections import deque
        fs = frames[start:start+len(orig)]
        F = {'A':deque(), 'B':deque(), 'C':deque()}
        for f in fs:
            F[f.id].append(f)
        new_fs = []
        for id in trans:
            f = F[id].popleft()
            new_fs.append(f)
        print "###", orig, trans, ''.join([f.id for f in new_fs])
        new_fs = frames[:start] + new_fs + frames[start+len(orig):]
        return new_fs
    s = ''.join( [f.id for f in frames] )
    print "... applying transformations ..."
    for orig, trans in transformations:
        start = 0
        end = len(s)
        while True:
            i = s.find(orig, start, end)
            if i < 0:
                break
            frames = apply_transform(frames, i, orig, trans)
            new_s = ''.join( [f.id for f in frames] )
            assert s[:i] + trans + s[i + len(trans):] == new_s
            print "changing %s >> %s" % (orig, trans)
            s = new_s
            start = i
    return frames

def create_frames(track):
    frames = flatten(track.frames_list)
    t0 = 0
    new_frames = []
    for frame in frames:
        new_frames.append(Frame(t0, frame, track.id, track.id))
        t0 += frame
    return new_frames

def show_frames(frames, show=False):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = numpy.fromiter((frame.start for frame in frames), float)
    ys = numpy.ones_like(xs)
    width = numpy.fromiter((frame.dur for frame in frames), float)
    bottom = numpy.fromiter((ord(frame.id) - ord('A') for frame in frames), float)
    C = {
        'A':'red', 'B':'blue', 'C':'green', 'D':'orange'
    }
    colors = [C[frame.id] for frame in frames]
    rects = ax.bar(xs, ys, width, bottom, color=colors, linewidth=1, aa=False)
    return (xs, ys, width, bottom, colors)
    if show:
        plt.show()

def show_frames2(frames, sep=0.2,ysize=2):
    from chaco.tools.api import PanTool, ZoomTool, DragTool
    from chaco.api import ArrayPlotData, Plot
    from chaco.base import n_gon
    # Enthought library imports
    from enable.api import Component, ComponentEditor
    from traits.api import HasTraits, Instance, Enum, CArray
    from traitsui.api import Item, Group, View
    from math import sqrt

    def _create_plot_component():
        pd = ArrayPlotData()
        polyplot = Plot(pd)
        for i, frame in enumerate(frames):
            #x = frame.start + frame.dur * 0.5
            #y = (ord(frame.id) - ord('A')) * 50
            #r = sqrt((frame.dur/2)**2 + (frame.dur/4)**2)
            #p = n_gon(center=(x, y), r=r, nsides=4, rot_degrees=45)

            x0 = frame.start
            y0 = (ord(frame.id) - ord('A')) * ysize
            x1 = x0 + frame.dur - sep
            y1 = y0 + ysize
            p = [(x1, y1), (x0, y1), (x0, y0), (x1, y0)]
            nx, ny = numpy.transpose(p)
            pd.set_data('x'+str(i), nx)
            pd.set_data('y'+str(i), ny)
            plot = polyplot.plot(('x'+str(i), 'y'+str(i)), type='polygon', hittest_type='poly')[0]
            #plot.tools.append(DataspaceMoveTool(plot, drag_button
        polyplot.tools.append(PanTool(polyplot))
        zoom = ZoomTool(polyplot, tool_mode="box", always_on=False)
        polyplot.overlays.append(zoom)
        return polyplot
    size = (800, 800)
    class Demo(HasTraits):
        plot = Instance(Component)

        traits_view = View(
                        Group(
                            Item('plot', editor=ComponentEditor(size=size),
                                 show_label=False),
                            orientation = "vertical"),
                        resizable=True
                        )

        def _plot_default(self):
             return _create_plot_component()

    d = Demo()
    d.configure_traits()

def varbpf(bpf):
    xs = numpy.arange(0, 1+0.001, 0.001)
    ys0 = map(bpf, xs)
    ys1 = list(accum(numpy.abs(numpy.diff(ys0))))
    xs = xs[:1000]
    assert len(ys1) == len(xs)
    return Linear(xs, ys1)

def calculate_frame_relative_pos(track):
    c = track.compression_curve
    dx = 1/len(track.frames)
    xs0 = numpy.arange(0, 1+dx, dx)
    #xs0 = map(track.normalized_pressence, xs0)
    xs0 = track.normalized_pressence.map(xs0)
    xs1 = numpy.abs(numpy.diff(xs0))
    xs2 = numpy.array(list( accum(dx * c(x) for dx, x in zip(xs1, xs0)) ))
    offsetL = xs0[0]
    offsetR = 1 - xs0[-1]
    xs3 = xs2 - offsetL
    xs3 = xs2 * ((1 - offsetL - offsetR) / max(xs2))
    xs3 += offsetL
    print ">>>", offsetL, xs3
    if any(x < 0 for x in xs3):
        1/0
    return xs3

def mix_frames():
    frames = []
    for track in tracks:
        rel_xs = calculate_frame_relative_pos(track)
        track.rel_frames = rel_xs
        frames.extend([frame._replace(start=rel_x) for rel_x, frame in zip(rel_xs, track.frames)])
    frames.sort()
    for i in range(100):
        new_frames = apply_transformations(frames, transformations)
        if frames == new_frames:
            break
        frames = new_frames
    new_frames = []
    t = 0
    for frame in frames:
        new_frames.append(frame._replace(start=t))
        t += frame.dur
    return new_frames

def make_measures(frames):
    measures = []
    for f in frames:
        m = Measure(f.start, f.dur, f.id, '___', '___')
        measures.append(m)
    return measures

def csv_index_frames(csvfile, outfile=None):
    if outfile is None:
        outfile = add_suffix(csvfile, 'IDX')
    counter = {'A':0, 'B':0, 'C':0}
    tuples, newrow = [], []
    for row in csvtools.read(csvfile):
        counter[row.id] += 1
        newrow.append(row.id + str(counter[row.id]))
        tuples.append(row)
    csvtools.writecsv(namedtuple_addcolumn(tuples, newrow, 'track_and_num'), outfile)

def buildtrack(track):
    """
    this is to be called after inittrack
    """
    print "------------------- processing track ", track.id
    is_order_reverse = (track.sections_order in ('descending', '>'))
    section_mindur = track.get('section_mindur')
    if section_mindur is None:
        section_mindur = track.num_sections * ifib1(0.5, track.mindur, track.maxdur)
    print "creating sections"
    track.sections_dur = create_sections(track)
    track.sections_hom = create_hom(track.hom_curve, track.sections_dur)
    print "generating frames list"
    track.frames_list  = generate_frames(track)
    print "creating individual frames"
    track.frames = create_frames(track)


# <<< DO >>> --------------------------------------------------------------------------
tracks = (A, B, C, D)
frames = []
for track in tracks:
    inittrack(track)
    buildtrack(track)

if 1:
    print "mixing frames"
    frames = mix_frames()
    numframes = sum(len(track.frames) for track in tracks)

    for track in tracks:
        track.ratio = sum(frame.dur for frame in track.frames) / sum(frame.dur for frame in frames)
        # track

    print "making measures"
    measures = make_measures(frames)
    csvtools.writecsv(measures, 'MEASURES')