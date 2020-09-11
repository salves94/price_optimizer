create table users(
    id integer primary key autoincrement,
    full_name text not null,
    email text not null unique,
    password text not null,
    admin boolean not null
);