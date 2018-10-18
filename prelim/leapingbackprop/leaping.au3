#include <Constants.au3>

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; AutoIt Version: 3.0																														;
; Language:       English																													;
; Platform:       Windows 7 Professional																									;
; Author:         Owen Morgan																												;
;																																			;
; SCRIPT FUNCTION																															;
;   Generates output from the Leaping Lagomorphs freeware program (http://pnb.mcmaster.ca/goldreich-lab/LL/Leaping_Lagomorphs.html),		;
;	which implements the Bayesian observer model (see references below) for perceptual length contraction in touch a la the tau effect 		;
; 	and sensory saltation (the "cutaneous rabbit"). It is recommended that inputs are generated from a normal distribution across a 		;
; 	physically realistic range and pasted into the empty file "LLtraininginputset1000.txt". When the script is run, they will be extracted	;
;	from this file, which should be saved in the current directory. If the script has trouble finding the file, uncomment line 55 and 		;
;	modify the first string argument in _FileReadToArray to the absolute path of the .txt file on your local machine and in this event, the	; 
;	same thing will have to be done on line 60. In any case, line 65 will have to be modified such that the string argument corresponds to 	;
;	the location of "Leaping Lagomorphs.exe" on your machine. LLtraininginputset1000example, LLtrainingoutputset1000example and all other	;
;	.csv and .txt in this same directory were pre-generated. If you are using the data for a backpropagation neural network, testing and 	;
;	validation sets	could also be produced with this script.																				;							;
;																																			;
;	Given that the Leaping Lagomorphs (LL) was designed using an unknown version of LabView, piping data directly to the text fields in 	;
;	the program's GUI could not be achieved (although it was attempted). Hence why I've resorted to using autoIt to manipulate the cursor	;
;	and keys directly. Consequently: (a) you can't use your machine while the script is running, (b) the script takes a long time to run	;
;	all 1000 input combinations. Furthermore, the Sleep delays used to account for loading between screens in LL were manually set based	;
; 	on my machine's speed of execution. You will most likely have to tune them to suit yours. 												;
;	-------------------------------------------------------------------------------------------------------------------------------------	;																																;
;																																			;
;	The Bayesian observer model:																											;
;		1. Goldreich, D. (2007). A Bayesian Perceptual Model Replicates the Cutaneous Rabbit and Other Tactile Spatiotemporal Illusions. 	;
;			PLoS ONE, 2(3), p.e333.																											;
;		2. Goldreich, D. and Tong, J. (2013). Prediction, Postdiction, and Perceptual Length Contraction: A Bayesian Low-Speed Prior 		;
;			Captures the Cutaneous Rabbit and Related Illusions. Frontiers in Psychology, 4.												;
;		3. Tong, J., Ngo, V. and Goldreich, D. (2016). Tactile length contraction as Bayesian inference. Journal of Neurophysiology, 		;
;			116(2), pp.369-379.																												;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

#include <File.au3>
#include <Array.au3>

Global $aCSV[1]
Global $intCount = 1
Global $intRow = 0
Global $intLineCount = 0

; Prompt the user to run the script - use a Yes/No prompt with the flag parameter set at 4 (see the help file for more details)
Local $iAnswer = MsgBox(BitOR($MB_YESNO, $MB_SYSTEMMODAL), "AutoIt Example", "Do you want to run the Leaping Lagomorphs test script?")
; Check the user's answer to the prompt (see the help file for MsgBox return values)
; If "No" was clicked (7) then exit the script
If $iAnswer = 7 Then
	MsgBox($MB_SYSTEMMODAL, "AutoIt", "OK.  Bye!")
	Exit
EndIf

$filePathInput = @WorkingDir & "LLtraininginputset1000.txt"
; $filePathInput = "C:\Users\Owen\Desktop\LLtraininginputset1000.txt"
_FileReadToArray($filePathInput, $aCSV, Default, ","); 
$intLineCount = _FileCountLines($filePathInput) - 1
Sleep(1000)
$filePathOutput = @WorkingDir & "LLtrainingoutputset1000.txt"
; $filePathOutput = "C:\Users\Owen\Desktop\LLtrainingoutputset1000.txt"

Run( "cmd /c " & " """ & $filePathOutput &  """ ", "", @SW_HIDE )
Sleep(1000)
; replace the path below with the location of the program on your local machine
Run("C:\Program Files (x86)\LL Project\Leaping Lagomorphs.exe")
WinWaitActive("Leaping Lagomorphs")
HotKeySet("{ESC}", "Terminate")

; START PROGRAM
Send("{Enter}")

While $intCount <= $intLineCount
$intCount = $intCount + 1

; ENTER NUMBER OF STIMULUS LOCATIONS
Send("{TAB}")
Send("3")
Send("{TAB}")
Send("{Enter}")
Sleep(1500)

; ENTER LOCATIONS (cm) AND # OF TAPS AT EACH LOCATION
MouseClick($MOUSE_CLICK_LEFT, 136, 438, 1, 0)
Send("{BACKSPACE}")
Send("0")
Send("{TAB}")
Send("{BACKSPACE}")
Send($aCSV[$intCount][1])
Send("{TAB}")
Send("{BACKSPACE}")
Send("{BACKSPACE}")
Send($aCSV[$intCount][1]*2)
Sleep(100)
MouseClick($MOUSE_CLICK_LEFT, 246, 442, 1, 0)
Send("{BACKSPACE}")
Send($aCSV[$intCount][2])
Send("{TAB}")
Send("{BACKSPACE}")
Send($aCSV[$intCount][2])
Send("{TAB}")
Send("{BACKSPACE}")
Send($aCSV[$intCount][2])
MouseClick($MOUSE_CLICK_LEFT, 143, 257, 1, 0)
Sleep(1500)

; REGULAR STIMULUS TIMING IS OKAY 
Send("{TAB}")
Send("{TAB}")
Send("{Enter}")
Sleep(1200)

; ENTER ISI (s)
MouseClick($MOUSE_CLICK_LEFT, 147, 426, 1, 0)
Send("{BACKSPACE}")
Send("{BACKSPACE}")
Send("{BACKSPACE}")
Send($aCSV[$intCount][0])
Send("{TAB}")
Send("{TAB}")
Send("{Enter}")
Sleep(1500)

; BALANCED SPATIAL ATTENTION IS OKAY 
Send("{Enter}")
Sleep(1500)

; ENTER SPATIAL ATTENTION sigma(s) (cm)
Send("{TAB}")
Send("{BACKSPACE}")
Send($aCSV[$intCount][3])
Send("{TAB}")
Send("{TAB}")
Send("{Enter}")
Sleep(1500)

; ENTER SPEED UNCERTAINTY sigma(v) (cm/s)
Send("{TAB}")
Send("{TAB}")
Send("{BACKSPACE}")
Send($aCSV[$intCount][4])
Send("{TAB}")
Send("{Enter}")
Sleep(1500)

; DEFAULT MEASUREMENT GENERATION OPTIONS
Send("{Enter}")
Sleep(1200)
MouseClick($MOUSE_CLICK_LEFT, 1160, 480, 3, 1)
Sleep(500)


; OPEN NOTEPAD LOG
Send("^c")
Sleep(100)
Send("{ALT DOWN}")
Send("{TAB}")
Send("{ALT UP}")
Sleep(100)
Send("^v")
Send("{Enter}")
Sleep(100)
Send("{ALT DOWN}")
Send("{TAB}")
Send("{ALT UP}")
Sleep(100)

; START AGAIN
MouseClick($MOUSE_CLICK_LEFT, 147, 259, 1, 2)
Sleep(1500)
MouseClick($MOUSE_CLICK_LEFT, 147, 259, 1, 10)
Sleep(2000)


WEnd

Func Terminate()
    Exit 0
EndFunc

; Finished!
