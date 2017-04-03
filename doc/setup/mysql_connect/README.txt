
You need mysql jdbc client available here (requires free oracle account sign-up):
https://dev.mysql.com/downloads/connector/j/
OR also contained in the attached zip. Put the .jar file somewhere to your liking on your local machine.

In matlab window type:
    edit(fullfile(prefdir,'javaclasspath.txt'))

In this file specify the full (platform formatted) path to the jdbc .jar file from above, for example:
/Users/pwatkins/Documents/MATLAB/tools/java/mysql-connector-java-5.1.39-bin.jar
OR
C:\Users\Paul\stuff\mysql-connector-java-5.1.39-bin.jar

You can add comments to this file using #

Save this file and restart matlab. Then run the .m file that is contained in the attached zip:
    mysql_connect

This should display the output:
    query "SELECT name from dumbTable" returned johndoe

For python, install the connector:
    pip install mysql-connector-python
OR download from https://dev.mysql.com/downloads/connector/python/
    python setup.py install
    
