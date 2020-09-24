create table users(
    id integer primary key autoincrement,
    full_name text not null,
    email text not null unique,
    password text not null,
    admin boolean not null
);

create table products(
    id integer primary key autoincrement,
    name text not null,
    sku text not null unique,
    description text
);