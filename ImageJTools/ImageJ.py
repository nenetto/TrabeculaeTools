import subprocess

class imagej(object):

    def __init__(self, imagej_exe):
        try:
            assert imagej_exe != ''
        except:
            print "[ERROR]: ImageJ path is empty"
        self._executable = imagej_exe

    def runImageJMacro(self, macroPath, macroParams, macroName = 'Macro'):
        '''Run a ImageJ Macro using command line

        Keyword arguments:
        macroPath                      -- Path to macro file *.ijm (mandatory)
        macroParams                    -- Parameters that the macro uses (mandatory)
        msgOpt [\'Running Macro ...\'] -- Message for debugging the will be shown when macro is running

        '''

        cmdline = self._executable + ' --no-splash -macro  \"' + macroPath + '\" \"' + macroParams + '\"'
        #print '    - Launching: ' + cmdline

        try:
            retcode = 0
            retcode = subprocess.call(cmdline, shell=True)
            if retcode == 0:
                pass #print '    - ' + macroName + " finished [OK]"
            else:
                print '    - [ERROR] ' + macroName + " finished with code [Handled Error Code]: ", retcode
        except OSError as e:
            print >> sys.stderr, '    - [ERROR] ' + macroName , " Execution Macro ImageJ failed:", e
