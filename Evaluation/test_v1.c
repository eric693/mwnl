#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

int main(int argc, char *argv[]) {
    // Initialize the Python interpreter.
    Py_Initialize();

    // Start measuring time.
    struct timeval begin, end;
    gettimeofday(&begin, NULL);

    // Open the Python file.
    FILE* PythonScriptFile = fopen("c_2_decoder_qpsk_gaussian_test.py", "r");
    if(PythonScriptFile == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Run the Python script.
    PyRun_SimpleFile(PythonScriptFile, "c_2_decoder_qpsk_gaussian_test.py");

    // Close the file.
    fclose(PythonScriptFile);

    // Stop measuring time and calculate the elapsed time.
    gettimeofday(&end, NULL);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;

    // Print the elapsed time.
    printf("C call pytorch time elapsed: %f seconds.\n", elapsed);

    // Clean up and close the Python interpreter.
    Py_Finalize();

    return 0;
}
