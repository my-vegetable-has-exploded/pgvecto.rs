su postgres
export PATH=$PATH:/usr/lib/postgresql/15/bin 
export PGDATA=/var/lib/postgresql/pg 
pg_ctl initdb
/usr/lib/postgresql/15/bin/pg_ctl -D /var/lib/postgresql/pg -l logfile start
/usr/lib/postgresql/15/bin/pg_ctl -D /var/lib/postgresql/pg stop