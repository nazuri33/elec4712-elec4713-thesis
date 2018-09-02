#include <Constants.au3>

;
; AutoIt Version: 3.0
; Language:       English
; Platform:       Win9x/NT
; Author:         Jonathan Bennett (jon at autoitscript dot com)
;
; Script Function:
;   Plays with the calculator.
;

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

_FileReadToArray("C:\Users\Owen\Desktop\LLtraininginputset1000.txt", $aCSV, Default, ","); 
$intLineCount = _FileCountLines("C:\Users\Owen\Desktop\LLtraininginputset1000.txt") - 1
Sleep(1000)


$d = "C:\Users\Owen\Desktop\LLtrainingoutputset1000.txt"
Run( "cmd /c " & " """ & $d &  """ ", "", @SW_HIDE )
Sleep(1000)

Run("C:\Program Files (x86)\LL Project\Leaping Lagomorphs.exe")
WinWaitActive("Leaping Lagomorphs")
HotKeySet("{ESC}", "Terminate")

; Wait for the calculator to become active. The classname "CalcFrame" is monitored instead of the window title
; WinWaitActive("[CLASS:CalcFrame]")

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

; Now quit by sending a "close" request to the calculator window using the classname
WinClose("[CLASS:CalcFrame]")

; Now wait for the calculator to close before continuing
WinWaitClose("[CLASS:CalcFrame]")

Func Terminate()
    Exit 0
EndFunc

; Finished!
