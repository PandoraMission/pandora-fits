# File Formats

File formats for Pandora FITS files are defined in these excel files. Excel files are used because they are highly human readable, and each sheet can be used to define each extension.

There are two files with definitions:

## Extension Types Files

For each detector and each level of data there is a file defining the extension types, for example

| Extension | Type       |
|-----------|------------|
| 0         | PrimaryHDU |
| 1         | ImageHDU   |
| 2         | ImageHDU   |
| 3         | ImageHDU   |
| 4         | ImageHDU   |
| 5         | TableHDU   |

This is used to check that input files have the right types of FITS extensions.

## Header Formats

For each detector and level the headers are defined in excel files. Each excel file has one sheet per extension. Headers are defined with the keyword name, the value expected (if any) and a comment.

| Name     | Value        | Comment                       |
|----------|--------------|-------------------------------|
| SIMPLE   | TRUE         | conforms to FITS standard     |
| BITPIX   | 8            | array data type               |
| NAXIS    | 0            | number of array dimensions    |
| EXTEND   |              |                               |
| EXTNAME  | PRIMARY      | name of extension             |
| NEXTEND  | 5            | number of standard extensions |
| SIMDATA  |              | simulated data                |
| SCIDATA  |              | science data                  |
| TELESCOP | NASA Pandora | telescope                     |
| INSTRMNT | NIRDA        | instrument                    |

This tool will check to ensure that all the correct header keys exist in the files, and if they are valued in the excel sheet the tool will check that they have the expected value.

## Dummy Files

`pandora-fits` can make dummy files with the formats described in this repository which have data of the correct `dtype` in them, but the data is all values of `1`.
