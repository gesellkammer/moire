(
/*
This patch has too functions:

A) translates the signal of the sax to a noise, following the
spectral contour of the original signal and boosting it

B) forwards all midi pedals to ~midi_destination, merging
both sax pedals into one (the talkpedal) (no need to
use MIDI Patchbay anymore).

*/

/************* CONFIGURATION *************
 ****************************************/

// channels begin with 1
~sax_inchan  = 1;
~sax_outchan = [1, 2];

~sax_noisepedal = 64;
~sax_talkpedal = 4;

~forward_midi = true;

// look for destinations: MIDIClient.destinations
~midi_destination = ["IAC Driver", "Bus 1"];

/************* END CONFIGURATION *********
 *****************************************/

~sax_inchan = ~sax_inchan - 1;
~sax_outchan = ~sax_outchan - 1;

~compress = {|sig, thresh= -24, ratio=2, attack=0.005, release=0.1|
    Compander.ar(sig, sig, thresh.dbamp, slopeBelow:1, slopeAbove: (1/ratio),
        clampTime:attack, relaxTime:release);
};

~check_bad = { |sig|
    Select.ar(CheckBadValues.ar(sig, 0, 0), [sig, DC.ar(0), DC.ar(0), sig]);
};

~compressorlimiter = {|sig, compthr= -3, limit_thresh= -0.1, ratio=10, attack=0.0008, release=0.5, lookahead=0.005|
    var look = max(lookahead, ControlDur.ir);
    //var compress = ~compress.(sig, soft_thresh, knee, ratio, attack:attack, release:release, autogain:0);
    var compress = ~compress.(sig, compthr, ratio:ratio, attack:attack, release:release);
    var limit = Limiter.ar(compress, limit_thresh.dbamp, look);
    // limit = ~check_bad.(limit);
    limit;
};

~whisper = {|sig, noise, bw=0.5|
    /*
    var bark_edges = #[0, 100, 200, 300, 400, 510, 630, 770, 920,
            1080, 1270, 1480, 1720, 2000, 2320, 2700,
            3150, 3700, 4400, 5300, 6400, 7700, 9500,
            12000];
    var bark_centers =  #[50, 150, 250, 350, 450, 570, 700, 840,
            1000, 1170, 1370, 1600, 1850, 2150, 2500,
            2900, 3400, 4000, 4800, 5800, 7000, 8500,
            10500];
    */
    /*
    var bark_edges = #[0, 200, 300, 400, 510, 770, 920,
        1080, 1270, 1480, 1720, 2000, 2320, 2700,
        3150, 3700, 4400, 5300, 6400, 9500,
        12000];
    var bark_centers =  #[100, 250, 350, 450, 640, 840,
        1000, 1170, 1370, 1600, 1850, 2150, 2500,
        2900, 3400, 4000, 4800, 5800, 8500,
        10500];
    */
    var bark_edges = #[0, 200, 400, 510, 770, 920,
        1080, 1480, 1720, 2000, 2320, 2700,
        3150, 3700, 4400, 5300, 6400, 9500,
        12000];
    var bark_centers =  #[100, 300, 450, 640, 840,
        1000, 1280, 1600, 1850, 2150, 2500,
        2900, 3400, 4000, 4800, 5800, 8500,
        10500];

    var chain = FFT(LocalBuf(2048), sig, hop:0.25);
    var powers = FFTSubbandPower.kr(chain, bark_edges, square:0, scalemode:1);
    var bands = BBandPass.ar(noise, bark_centers, bw:bw) * powers;
    Mix.ar(bands);
};

~selectx2 = {|which, a0, a1|
    XFade2.ar(a0, a1, which*2-1);
};

fork {
    var inchan = ~sax_inchan;
    var outchan = ~sax_outchan;
    ~saxsyn = { |noisegain=16, noisewet=0.6, whispgain=20, whispwet=0.8, whispbw=0.5,
                 att_ms=330, rel_ms=150, compthr= -3, limthr= -0.2|
        var bypass = \bypass.kr(0, 0.2, fixedLag:true);
        var a0 = SoundIn.ar(inchan);
        var noise0 = PinkNoise.ar();
        var a0_env = Amplitude.ar(a0, attackTime:att_ms/1000, releaseTime:rel_ms/1000);
        var noise = noise0 * (a0_env * noisegain);
        var whispered = ~whisper.(a0, noise0, whispbw) * whispgain;
        var signal = ~selectx2.(whispwet, a0, whispered);
        var out = ~selectx2.(noisewet, signal, noise);
        out = ~compressorlimiter.(out, compthr:compthr, limit_thresh:limthr);
        out = ~selectx2.(bypass, a0, out);
        Out.ar(outchan, out);
    }.play;
    s.sync;

    MIDIClient.init;

    // Connect to all but IAC
    if (MIDIClient.initialized.not) { MIDIClient.init };
    MIDIClient.sources.do({ |src,i|
        if( src.device != "IAC Driver" ) {
            "connecting to %".format(src.device).postln;
            MIDIIn.connect(i,src);
        };
    });

    2.wait;

    m = MIDIOut.newByName(~midi_destination[0], ~midi_destination[1]);
    if( ~forward_midi ) {
        MIDIFunc.cc({|val, cc, chan|
            if( cc != ~sax_noisepedal ) {
                m.control(chan, cc, val);
            };
        });
    };

    MIDIFunc.cc(ccNum:~sax_noisepedal, func:
        { |val, cc, chan|
            var bypass = (val > 0).asInteger;
            m.control(chan, ~sax_talkpedal, val);
            ~saxsyn.set(\bypass, bypass);
        }
    );
}
)

// ----------------------------------------------
~saxsyn.autogui
