Custom simulations for integration into Simbrain.jar source code. Refer to the notes below for the compilation process involved. Note that all paths referenced are in reference to my local machine and these instructions are for personal reference.

1. All modifications to the stimulus input workspace need to be made to the file SORNModWithAbridgingSimplified.zip in 
   E:/Setups/Simbrain 3.03/scripts/scriptmenu so that the current openWorkspace call in my SORNAbridging.java custom class works.
2. To update custom class, modify SORNAbridging.java in 
   .../Simbrain 3.03/github_repo/simbrain/src/org/simbrain/custom_sims/simulations/sorn_abridging. You probably want to test changes on      the bash script first (SORNModWithAbridgingStimulus.bsh) since it’s easier to run considering no compilation and .jar integration          script has been written (yet).  
3. Then add the changes to the .java file, recompile with javac -classpath "../../../../../../../../Simbrain.jar" *.java 
   from simulations/sorn_abridging in Git Bash, add the newly compiled class to Simbrain.jar in WinRAR, and reopen SimBrain. 
