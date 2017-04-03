
driverClassName = 'com.mysql.jdbc.Driver';
classLoader = com.mathworks.jmi.ClassLoaderManager.getClassLoaderManager;
%driverClass = classLoader.loadClass(driverClassName);
classLoader.loadClass(driverClassName);

% jdbc:mysql://[host][,failoverhost...]
% [:port]/[database]
% [?propertyName1][=propertyValue1]
% [&propertyName2][=propertyValue2]

connStr = 'jdbc:mysql://ndsw-cdcu-yello:3306/dbCDCUtest';
con = java.sql.DriverManager.getConnection(connStr,'adminCDCU','schmee#schm44');

query = 'SELECT name from dumbTable';

stmt = con.createStatement();
rs = stmt.executeQuery(query);
if ~rs.next()
    fprintf(1, '\nError, bad database connection\n'); stmt.close(); return;
end

name = char(rs.getString('name'));
stmt.close();
display(sprintf('query "%s" returned %s',query,name));
con.close();
