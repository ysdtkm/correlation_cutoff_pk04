# Correlation-cutoff method for Covariance localization in Strongly coupled DA
* Source codes used for T. Yoshida and E. Kalnay (2018, MWR, doi:10.1175/MWR-D-17-0365.1)

## Usage
* Just type "make" (GNU make will resolve the dependency,
detect updated source files, and execute minimal necessary commands).
* Most of the experimental parameters can be edited in Py/const.py

## Reproducibility Note
* The figures on the paper are calculated by the commits specified below
    * fe8da36 Figure 2
    * 3785478 Figure 3, 4-member
    * b833893 Figure 3, 6-member
    * 3a19d2f Figure 3, 10-member
    * eff4dc1 Attachment for reviewers, 100-member
    * 596f497 Attachment for reviewers, mean background covariance
* Experiments above are conducted on *halo* server of AOSC, UMD
    * AMD Opteron(TM) Processor 6272
    * Red Hat Enterprise Linux Server release 7.5 (Maipo)
    * GNU Bash 4.2.46
    * Python 3.6.2
        * numpy 1.13.1
        * scipy 0.19.1
        * matplotlib 2.0.2
