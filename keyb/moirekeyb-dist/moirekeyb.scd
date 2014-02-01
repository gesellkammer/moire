(
/*

Instructions
============

1. OSX --> Audio and MIDI Setup: Select your audio interface
2. Connect your MIDI keyboard
3. Configure the CC values if necessary (see ~cc)
4. Configure other values (see ~config)
5. Press COMMAND-Enter to start
6. After a few seconds a window will pop up

To stop sound, press COMMAND-. (Command-Period) or close the window

*/


// ----------------------- CONFIGURATION --------------------------


~cc = (
    volpedal: 7,
    modwheel: 1,
    panic: nil,        // if set, when this controller is touched all midi notes will be stopped
);

~config = (
    a4: 442,
    velocity_when_insensitive: 90,
    saw_threshold:  48,
    pitchbend_cents: 50,
    modwheel_cents: 100,
    limiter_soft_thresh: -3,
    settings_filename: "moirekeyb-settings.txt"
);


// ------------------------ END CONFIGURATION -----------------------

~defaults = (
    'note_attack_ms':  10,
    'note_release_ms': 50,
    'volpedal': 127,       // 0-127
    'mastervolume': 1,       // 0-1
    'notemaxgain_db': -6,
    'sensitive_keyboard': true,
    'sawduty':1,
);

~getcwd = {
    var rout = Routine({1}).play(AppClock);
    rout.executingPath.dirname;
};

~env_fromxy = { |xs, ys, curve='lin'|
    var deltas = (xs.size - 1).collect {|i|
        xs[i+1] - xs[i]
    };
    Env.new(ys, deltas, curve);
};

~bpflinear = { |...seq|
    var xs, ys;
    # xs, ys = seq.clump(2).flop;
    ~env_fromxy.(xs, ys);
};

~compress = {|sig, thresh= -24, knee=8, ratio=2, gain= -3, autogain=1, attack=0.0, release=0.1|
    /*
    -- the softknee begins at thresh - knee
    -- gain has only meaning when autogain is on
    -- a bigger ratio means more compression compression
    */
    var times = [attack, release];
    var slopeAbove = 1.0/ratio;
    var gain2 = Select.kr(autogain, [gain, (thresh.neg * ( 1 - slopeAbove )) + gain]); // autogain
    var amp = Amplitude.ar( sig, *times ).linlin(
        (thresh-knee).dbamp, (thresh+knee).dbamp, 0, 1
    ).clip(0,1).sqrt;
    knee = knee.max(0.0001);
    // slopeAbove = 1.blend( slopeAbove, amp );
    // thresh = (thresh-knee).blend( thresh, amp );
    slopeAbove = amp.linlin(0, 1, slopeAbove, 1);
    thresh     = amp.linlin(0, 1, thresh, thresh-knee);
    Compander.ar( sig, sig, thresh.dbamp, 1, slopeAbove, *times ) * gain2.dbamp;
};

~check_bad = { |sig|
    Select.ar(CheckBadValues.ar(sig, 0, 0), [sig, DC.ar(0), DC.ar(0), sig]);
};

~limiter = {|sig, soft_thresh= -3, hard_thresh= -0.1, knee=3, ratio=4, attack=0.0008, release=0.5, lookahead=0|
    var look = max(lookahead, ControlDur.ir);
    var compress = ~compress.(sig, soft_thresh, knee, ratio, attack:attack, release:release, autogain:0);
    var limit = Limiter.ar(compress, hard_thresh.dbamp, look);
    limit = ~check_bad.(limit);
    limit;
};

// -------------- PERSISTENCE ------------------

~config.settings_path = ~getcwd.() +/+ ~config.settings_filename;

~settings_manager = {|path, persist_period|
    var data;
    var persist_task;
    var load = {|path|
        "reading settings from %".format(path).postln;
        data = if(File.exists(path)) {
            var file, str, dict;
            file = File(path, "r");
            str = file.readAllString;
            "settings: %".format(str).postln;
            dict = str.interpret;
            if ( dict.isKindOf(Dictionary).not ) {
                "settings: problem reading the saved settings".postln;
                ();
            } {
                dict;
            };
        }{
            ();  // <---- defaults to an empty Event
        };
        "finished reading settings".postln;
        data;
    };
    var save = {|data, path|
        var file = File(path, "w");
        var str = data.asCompileString;
        "saving settings".postln;
        file.write(str);
        file.close;
    };

    var persist_start = {|data, path, period|
        persist_task = Routine {
            loop {
                save.(data, path);
                period.wait;
            }
        }.play(AppClock);
        persist_task;
    };

    var persist_stop = {
        save.(data, path);
        persist_task.stop;
        persist_task = nil;
    };

    persist_period = persist_period ? 10;
    path = path.absolutePath;
    data = load.(path);

    if( File.exists(path.dirname).not ) {
        File.mkdir(path.dirname);
    };

    CmdPeriod.doOnce({
        {
            save.(data, path);
        }.defer;
    });

    persist_task = persist_start.(data, path, persist_period);

    (data:data, persist_task:persist_task, stop:persist_stop, save:save, load:load);
}.(~config.settings_path);

// Fill settings with defaults
~settings = ~settings_manager.data;
~defaults.pairsDo {|k, v|
    ~settings[k] = ~settings[k] ? v;
};

"Settings: ".postln;
~settings.postln;

// -----------------------------------------------------------------------------

fork {
    var postgroup, outsynth;
    var bend_offset = 0;
    var global_bend = 0;
    var notes = Array.fill(127, nil);
    var pitchbend_amount = ~config.pitchbend_cents / 100;
    var modwheel_amount = ~config.modwheel_cents / 100;
    var settings = ~settings;
    var bus, notegroup;

    s.options.numInputBusChannels = 0;
    s.options.numOutputBusChannels = 2;
    s.reboot;

    4.wait;

    s.sync; // ------------------------------------------------------------------

    bus = Bus.audio(s);
    notegroup = Group();

    SynthDef(\note, {|outbus, midinote, velocity, gate=1, bend=0, bend_offset=0, i_maxgain=0, i_attack=0.01, i_release=0.1, sawduty=1|
        var midi = (midinote + bend + bend_offset);
        var a4 = ~config.a4;
        var freq = ( 2**((midi - 69)/12.0) * a4 ).lag(0.2);
        var vel2amp_curve = ~bpflinear.(
            0,  -90,
            32, -32,
            64, -6,
            127, 0
        );
        var amp = IEnvGen.kr(vel2amp_curve, velocity).dbamp * (i_maxgain.dbamp);
        var sinus = SinOsc.ar(freq);
        //var saw = Saw.ar(freq);
        //var saw = SyncSaw.ar(freq, freq*1.2);
        // var saw = LFSaw.ar(freq);
        var saw = VarSaw.ar(freq, width:sawduty); // + Saw.ar(freq) * 0.707;
        var env = Env.adsr(attackTime:i_attack, sustainLevel:1, releaseTime:i_release);
        var gen = EnvGen.ar(env, gate, doneAction:2) * amp;
        var out = Select.ar(midinote < ~config.saw_threshold, [sinus, saw]) * gen;
        Out.ar(outbus, out);
    }).add;

    s.sync; // --------------------------------------------------

    postgroup = Group.after(~notegroup);
    outsynth = {|inbus, outbus, volpedal=127, mastervolume=1|
        var curve = ~bpflinear.(
            0,  -90,
            32, -32,
            64, -6,
            127, 0
        );
        var amp = (IEnvGen.kr(curve, volpedal).dbamp * mastervolume).clip(0, 1).lag(0.2);
        var a0 = In.ar(bus) * amp;
        var out = ~limiter.(a0, soft_thresh:~config.limiter_soft_thresh);
        Out.ar([0, 1], out);  // L & R chan. are the same
    }.play(postgroup, args:[inbus:bus, outbus:0, volpedal:settings.volpedal, mastervolume:settings.mastervolume]);

    s.sync; // --------------------------------------------------

    MIDIClient.init;
    MIDIIn.connectAll;

    s.sync;

    /***********************************
                  MIDI
    ************************************/

    // volpedal
    MIDIFunc.cc({|val, cc, chan|
        outsynth.set(\volpedal, val);
        settings.volpedal = val;
    }, ~cc.volpedal);

    // pitchwheel
    MIDIFunc.bend({|val|
        var bend = val.linlin(0, 16383, pitchbend_amount.neg, pitchbend_amount);
        global_bend = bend;
        notegroup.set(\bend, bend);
    });

    // modwheel
    MIDIFunc.cc({|val, cc, chan|
        var bend = val.linlin(0, 127, 0, modwheel_amount);
        notegroup.set(\bend_offset, bend);
        settings.bend_offset = bend;
    }, ccNum: ~cc.modwheel);

    // noteon & noteoff
    MIDIFunc.noteOn({|vel, midinote|
        var velocity = ~settings.sensitive_keyboard.if(vel, 90);
        var note = Synth(\note, target:notegroup, args:
            [
                outbus:bus, midinote:midinote, velocity:velocity,
                bend:global_bend, bend_offset:settings.bend_offset,
                i_maxgain: ~settings.notemaxgain_db,
                i_attack:  ~settings[\note_attack_ms]/1000, i_release:~settings[\note_release_ms]/1000,
                sawduty: ~settings.sawduty
        ]);
        notes[midinote] = note;
        // defer { ~scope.cycle = 44100/midinote.midicps; };
    });

    MIDIFunc.noteOff({|vel, midinote|
        notes[midinote].set(\gate, 0);
        notes[midinote] = nil;
    });

    // panic
    if( ~cc.panic.notNil ) {
        MIDIFunc.cc({|val|
            notes.do {|syn|
                if(syn.notNil) {
                    syn.set(\gate, 0);
                };
            };
            notes.fill(nil);
        }, ccNum: ~cc.panic);
    };

    // --------------------- GUI -------------------------
    defer {
        var width = 500;
        var height = 500;
        var lineheight = 20;
        var default_fontsize = 13;
        var win = Window.new(name:"moire", bounds:width@height).front;
        win.view.decorator = FlowLayout(win.bounds, 20@10, 10@10);
        CmdPeriod.doOnce {
            if( win.isClosed.not ) {
                win.close;
            };
        };
        win.onClose = {
            s.freeAll;
            OSCdef.freeAll;
            MIDIdef.freeAll;
            ~scope.free;
            ~levels.free;
            if( ~settings_manager.persist_task.notNil ) {
                ~settings_manager.persist_task.stop;
            };

        };
        ~newslider = {|param, label, spec, defaultvalue, sendtonote=false|
            var settings = ~settings;
            var value0 = settings[param] ? defaultvalue ? 0;
            var func;
            value0 = if(value0.isNaN) {0} {value0};
            func = {|slider|
                var v = slider.value;
                settings[param] = v;
                if( sendtonote ) {
                    notegroup.set(param, v);
                };
            };
            EZSlider(win, (width*0.9)@lineheight, " % ".format(label), spec, labelWidth:(width*0.25), numberWidth:60, action:func, initVal:value0)
            .round_(0.01)
            .font_(Font("Helvetica", default_fontsize));
        };
        ~newslider.(\note_attack_ms, "Attack", [0, 100], 10);
        ~newslider.(\note_release_ms, "Release", [0, 500], 50);
        ~newslider.(\sawduty, "Saw Duty", [0, 1], 0.01, sendtonote:true);
        ~newslider.(\notemaxgain_db, "Note Volume (dB)", [-40, 0], ~settings.notemaxgain_db);

        EZSlider(win,
            (width*0.9)@lineheight, " % ".format("Master Volume (dB"), [-24, 0],
            labelWidth:(width*0.25), numberWidth:60, initVal:settings.mastervolume.ampdb,
            action: {|slider|
                var amp = slider.value.dbamp;
                settings.mastervolume = amp;
                outsynth.set(\mastervolume, amp);
            }
        );

        ToggleButton(win, "Keyboard Sensitivity", minWidth:width*0.9, minHeight:40, init:~settings.sensitive_keyboard,
            onFunction:{|but, value|
                ~settings.sensitive_keyboard = value;
            }
        ).font_(Font("Helvetica", default_fontsize));

        ~scope = Stethoscope(s, numChannels:1, index:0, view:win.view);
        //~levels = ServerMeter(s);
        ~levels = ServerMeterView(s, win, 0@0, 2, 2);
        //~levels.view.keyDownAction = nil;

    };
}
)
