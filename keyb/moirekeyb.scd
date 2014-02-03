(
/*

Instructions
============

1. OSX --> Audio and MIDI Setup: Select your audio interface
2. Connect your MIDI keyboard
3. Configure the CC values if necessary (see ~cc)
4. Configure other values (see ~config)
5. Press COMMAND-B to to start DSP
6. Press COMMAND-Enter to start the patch itself.
7. A window will pop up

To stop sound, close the window.

Troubleshooting
===============

Q: The MIDI Keyboard is not working
A: It was probably connected after Supercollider was started.
   Go to menu "Language/Reboot Interpreter"
   Wait until you see "compile done" in the Post Window.
   Press COMMAND-Enter to start.
   At the Post Window you will see the MIDI devices detected (look for
   "MIDI sources"

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
    settings_filename: "moirekeyb-settings.txt"
);


// ------------------------ END CONFIGURATION -----------------------
//
//
// ------------------  From here on, DO NOT TOUCH  ------------------

~defaults = (
    note_attack_ms:  10,
    note_release_ms: 50,
    volpedal: 127,       // 0-127
    mastervolume: 1,       // 0-1
    notemaxgain_db: -6,
    sensitive_keyboard: true,
    sawduty:1,
    pitchbend_cents: 50,
    modwheel_cents: 100,
    comp_thresh: -3,
    comp_ratio: 10,
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

~compress2 = {|sig, thresh= -24, knee=8, ratio=2, gain= -3, autogain=1, attack=0.0, release=0.1|
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

~compress = {|sig, thresh= -24, ratio=2, attack=0.005, release=0.1|
    Compander.ar(sig, sig, thresh.dbamp, slopeBelow:1, slopeAbove: (1/ratio),
        clampTime:attack, relaxTime:release);
};

~check_bad = { |sig|
    Select.ar(CheckBadValues.ar(sig, 0, 0), [sig, DC.ar(0), DC.ar(0), sig]);
};

~compressorlimiter = {|sig, comp_thresh= -3, limit_thresh= -0.1, ratio=10, attack=0.0008, release=0.5, lookahead=0.005|
    var look = max(lookahead, ControlDur.ir);
    //var compress = ~compress.(sig, soft_thresh, knee, ratio, attack:attack, release:release, autogain:0);
    var compress = ~compress.(sig, comp_thresh, ratio:ratio, attack:attack, release:release);
    var limit = Limiter.ar(compress, limit_thresh.dbamp, look);
    limit = ~check_bad.(limit);
    limit;
};

~limiter2 = {|sig, hard_thresh= -0.1, lookahead=0.005|
    Limiter.ar(sig, hard_thresh.dbamp, lookahead);
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
    // var pitchbend_amount = ~config.pitchbend_cents / 100;
    // var modwheel_amount = ~config.modwheel_cents / 100;
    var settings = ~settings;
    var bus, notegroup;
    var pitchbend_display, volpedal_display;

    if( s.serverRunning.not ) {
        Server.killAll;
        2.wait;
        s.boot;
    };

	if( s.options.numInputBusChannels > 0 ) {
		s.options.numInputBusChannels = 0;
		s.options.numOutputBusChannels = 2;
		s.reboot;
		4.wait;
	};

    "--> Starting MIDI".postln;
    MIDIClient.init;
    MIDIClient.restart;
    2.wait;
    MIDIIn.connectAll;


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
    outsynth = Ref();

    ~create_outsynth = {
        var syn = {|inbus, outbus, volpedal=127, mastervolume=1, comp_thresh, comp_ratio|
            var curve = ~bpflinear.(
                0,  -90,
                32, -32,
                64, -6,
                127, 0
            );
            var amp = (IEnvGen.kr(curve, volpedal).dbamp * mastervolume).clip(0, 1).lag(0.2);
            var a0 = In.ar(bus) * amp;
            var out = ~compressorlimiter.(a0, comp_thresh:comp_thresh, ratio:comp_ratio);
            Out.ar([0, 1], out);  // L & R chan. are the same
        }.play(postgroup, args:[
            inbus:bus, outbus:0, volpedal:settings.volpedal, mastervolume:settings.mastervolume,
            comp_thresh:settings.comp_thresh, comp_ratio:settings.comp_ratio
        ]);
        if( outsynth.value.notNil ) {
            outsynth.value.free;
        };
        outsynth.value = syn;
    };
    ~create_outsynth.();

    s.sync; // --------------------------------------------------

    ~panic = {
        notegroup.freeAll;

        // ~create_outsynth.();
        notes.fill(nil);

    };

    // --------------------- GUI -------------------------
    defer {
        var width = 500;
        var height = 680;
        var lineheight = 18;
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

        win.view.keyDownAction = {|view, key|
            if( key == $p ) { ~panic.() };
        };

        ~newslider = {|param, label, spec, defaultvalue, sendtonote=false, sendtooutsynth=false, func, round=0.01|
            var settings = ~settings;
            var value0 = settings[param] ? defaultvalue ? 0;
            value0 = if(value0.isNaN) {0} {value0};
            func = func ? {|slider|
                var v = slider.value.round(round);
                settings[param] = v;
                case
                { sendtonote } { notegroup.set(param, v) }
                { sendtooutsynth } { outsynth.value.set(param, v) }
                ;
            };
            EZSlider(win, (width*0.9)@lineheight, " % ".format(label), spec, labelWidth:(width*0.25), numberWidth:60, action:func, initVal:value0)
            .round_(round)
            .font_(Font("Helvetica", default_fontsize));
        };

        ~display = {|label, spec, value0, round=0.01|
            var magn = 0.8;
            EZSlider(win, (width*0.9)@(lineheight*magn), " % ".format(label), spec, labelWidth:(width*0.25), numberWidth:60, initVal:value0)
            .round_(round)
            .setColors(sliderBackground:Color.black, knobColor:Color.green, stringColor:Color.grey)
            .font_(Font("Helvetica", default_fontsize*magn, bold:true));
        };

        ~newslider.(\note_attack_ms, "Attack", [0, 100], 10, round:1);
        ~newslider.(\note_release_ms, "Release", [0, 300], 50, round:1);
        ~newslider.(\sawduty, "Saw Duty", [0, 1], 0.01, sendtonote:true);
        ~newslider.(\pitchbend_cents, "Pitchbend Cents", [0, 200], ~settings.pitchbend_cents, round:1);
        ~newslider.(\modwheel_cents, "Modwheel Cents", [0, 200], ~settings.modwheel_cents, round:1);
        ~newslider.(\comp_thresh, "Comp Thresh (dB)", [-24, 0], round:0.1, sendtooutsynth:true);
        ~newslider.(\comp_ratio, "Comp Ratio", [1, 24], round:0.1, sendtooutsynth:true);

        ~newslider.(\notemaxgain_db, "Note Volume (dB)", [-24, 0], ~settings.notemaxgain_db, round:0.1);
        ~newslider.(\mastervolume, "Master Vol. (dB)", [-24, 0], ~settings.mastervolume.ampdb, round:0.1, func:
            { |slider|
                var amp = slider.value.dbamp;
                settings.mastervolume = amp;
                outsynth.value.set(\mastervolume, amp);
        });


        Button(win, width*0.9@40)
        .states_([
            ["Keyboard NOT sensitive", Color.black, Color.red],
            ["Keyboard sensitive", Color.white, Color.green]
        ])
        .value_(~settings.sensitive_keyboard.asInteger)
        .action_({|butt|
            ~settings.sensitive_keyboard = butt.value.booleanValue
        });

        pitchbend_display = ~display.("Pitchbend (Cents)", [-200, 200], 0, round:1);
        volpedal_display = ~display.("Volpedal", [0, 127], ~settings.volpedal, round:1);

        ~scope = Stethoscope(s, numChannels:1, index:0, view:win.view);
        ~levels = ServerMeterView(s, win, 0@0, 2, 2);

        Button(win, width*0.9@40)
        .states_([
            ["PANIC (Press P)", Color.white, Color.red]
        ])
        .value_(~settings.sensitive_keyboard.asInteger)
        .mouseDownAction_({
            ~panic.();
        });
    };

    0.1.wait;  // we need to wait for the defer

    /***********************************
                  MIDI
    ************************************/

    // volpedal
    MIDIFunc.cc({|val, cc, chan|
        outsynth.value.set(\volpedal, val);
        settings.volpedal = val;
        defer {
            volpedal_display.value = val;
        }
    }, ~cc.volpedal);

    // pitchwheel
    MIDIFunc.bend({|val|
        var pitchbend_amount = settings.pitchbend_cents/100;
        var bend = val.linlin(0, 16383, pitchbend_amount.neg, pitchbend_amount);
        global_bend = bend;
        notegroup.set(\bend, bend);
        defer {
            pitchbend_display.value = bend * 100;
        };
    });

    // modwheel
    MIDIFunc.cc({|val, cc, chan|
        var modwheel_amount = settings.modwheel_cents / 100;
        var bend = val.linlin(0, 127, 0, modwheel_amount);
        notegroup.set(\bend_offset, bend);
        settings.bend_offset = bend;
    }, ccNum: ~cc.modwheel);

    // noteon & noteoff
    MIDIFunc.noteOn({|vel, midinote|
        var velocity, note;
        if( vel > 0 ) {
            velocity = ~settings.sensitive_keyboard.if(vel, 90);
            note = Synth(\note, target:notegroup, args:
                [
                    outbus:bus, midinote:midinote, velocity:velocity,
                    bend:global_bend, bend_offset:settings.bend_offset,
                    i_maxgain: ~settings.notemaxgain_db,
                    i_attack:  ~settings[\note_attack_ms]/1000, i_release:~settings[\note_release_ms]/1000,
                    sawduty: ~settings.sawduty
            ]);
            notes[midinote] = note;
        } {
            // this should not happen!
            "noteon with vel. 0!".postln;
            notes[midinote].set(\gate, 0);
            notes[midinote] = nil;
        };
    });

    MIDIFunc.noteOff({|vel, midinote|
        notes[midinote].set(\gate, 0);
        notes[midinote] = nil;
    });

    // panic
    if( ~cc.panic.notNil ) {
        MIDIFunc.cc({|val|
            ~panic.();
        }, ccNum: ~cc.panic);
    };
}
)