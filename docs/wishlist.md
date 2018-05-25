- fit by batch (keras style)
- two different modes of operation:
    1. fit your data to a (to some degree) customizable vampnet
    2. take parts of the project (losses, ...?) to build it into the user's own network architecture
- design so that case 1. is independent on whether tf or pytorch is used and case 2 offer two different implementations
