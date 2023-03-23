#!/bin/bash
echo "start install";
function httpdinstall (){
	yum -y install httpd httpd-manual mod_ssl mod_perl mod_auth_mysql;
	systemctl start httpd.service;
}
function mysqldownload (){
	yum install -y libaio &
	touch list.txt;
	echo 'https://mirrors.aliyun.com/mysql/MySQL-5.7/mysql-community-server-5.7.36-1.el7.x86_64.rpm' >> list.txt;
	echo 'https://mirrors.aliyun.com/mysql/MySQL-5.7/mysql-community-client-5.7.36-1.el7.x86_64.rpm' >> list.txt;
	echo 'https://mirrors.aliyun.com/mysql/MySQL-5.7/mysql-community-common-5.7.36-1.el7.x86_64.rpm' >> list.txt;
	echo 'https://mirrors.aliyun.com/mysql/MySQL-5.7/mysql-community-libs-5.7.36-1.el7.x86_64.rpm' >> list.txt;
	echo 'https://mirrors.aliyun.com/mysql/MySQL-5.7/mysql-community-libs-compat-5.7.36-1.el7.x86_64.rpm' >> list.txt;
	echo 'http://dev.mysql.com/get/mysql57-community-release-el7-10.noarch.rpm' >> list.txt;
	wget -i list.txt;
}
function mysqlinstall (){
	yum remove mariadb-libs;
	yum -y install mysql-community-common-5.7.36-1.el7.x86_64.rpm;
	yum -y install mysql-community-libs-5.7.36-1.el7.x86_64.rpm;
	yum -y install mysql-community-libs-compat-5.7.36-1.el7.x86_64.rpm;
	yum -y install mysql-community-client-5.7.36-1.el7.x86_64.rpm;
	yum -y install mysql-community-server-5.7.36-1.el7.x86_64.rpm;
	rpm --import https://repo.mysql.com/RPM-GPG-KEY-mysql-2022;
	yum -y install mysql57-community-release-el7-10.noarch.rpm;
	yum -y install mysql-community-server-5.7.36;
	systemctl start mysqld.service;
}
function phpinstall(){
	yum -y install php php-mysql gd php-gd gd-devel php-xml php-common php-mbstring php-ldap php-pear php-xmlrpc php-imap;
	echo "<?php phpinfo(); ?>" > /var/www/html/phpinfo.php;
	wget https://labfileapp.oss-cn-hangzhou.aliyuncs.com/phpMyAdmin-4.0.10.20-all-languages.zip --no-check-certificate;
	yum install -y unzip;
	unzip phpMyAdmin-4.0.10.20-all-languages.zip -d /var/www/html;
	mv /var/www/html/phpMyAdmin-4.0.10.20-all-languages /var/www/html/phpmyadmin;
}
mysqldownload &
httpdinstall &
phpinstall &
wait;
mysqlinstall;
systemctl restart httpd;
pwline=$(grep "password" /var/log/mysqld.log);
pw=${pwline##*:};
echo ${pwline};
echo ${pw};
mysql -uroot -p --connect-expired-password <<EOF
set global validate_password_policy = 'LOW';
ALTER USER 'root'@'localhost' IDENTIFIED BY '12345678';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' IDENTIFIED BY '12345678';
FLUSH PRIVILEGES;
exit
EOF
echo "intall finished";
