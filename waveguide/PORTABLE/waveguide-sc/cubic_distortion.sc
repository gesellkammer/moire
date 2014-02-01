FaustCubicDistortion : UGen
{
  *ar { | in1, drive(0.0), offset(0.0) |
      ^this.multiNew('audio', in1, drive, offset)
  }

  *kr { | in1, drive(0.0), offset(0.0) |
      ^this.multiNew('control', in1, drive, offset)
  } 

  checkInputs {
    if (rate == 'audio', {
      1.do({|i|
        if (inputs.at(i).rate != 'audio', {
          ^(" input at index " + i + "(" + inputs.at(i) + 
            ") is not audio rate");
        });
      });
    });
    ^this.checkValidInputs
  }

  name { ^"FaustCubicDistortion" }
}

