DROP DATABASE IF EXISTS quantlabs_dev;
CREATE DATABASE quantlabs_dev;
GRANT ALL PRIVILEGES ON quantlabs_dev.* TO 'sync_user'@'%';
FLUSH PRIVILEGES;