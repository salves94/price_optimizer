create table "users" (
    "id" integer primary key autoincrement,
    "full_name" text not null,
    "email" text not null unique,
    "password" text not null
);

create table "products" (
    "id" integer primary key autoincrement,
    "name" text not null,
    "sku" text not null unique
);

create table "price_optimization_inputs" (
    "id" integer primary key autoincrement,
    "product_id" integer not null,
    "time_steps" integer not null,
    "max_price" real not null,
    "price_step" real not null,
    "q_0" real not null,
    "k" real not null,
    "unit_cost" real not null,
    "increase_coefficient" real not null,
    "decrease_coefficient" real not null,
    "gamma" real not null,
    "target_update" real not null,
    "batch_size" integer not null,
    "learning_rate" real not null,
    "num_episodes" integer not null,
    CONSTRAINT fk_products
        FOREIGN KEY (product_id)
        REFERENCES products(id)
        ON DELETE CASCADE
);

create table "price_optimization_results"(
    "id" integer primary key autoincrement,
    "price_optimization_inputs_id" integer not null UNIQUE,
    "constant_price" real not null,
    "constant_profit" real not null,
    "sequence_prices" text not null,
    "sequence_profit" real not null,
    "best_profit_results" text not null,
    "env_simulation_src" text not null,
    "optimal_seq_price_src" text not null,
    "returns_variation_src" text not null,
    "price_schedules_src" text not null,
    "td_errors_src" text not null,
    "correlation_src" text not null,
    CONSTRAINT fk_price_optimization_inputs
        FOREIGN KEY (price_optimization_inputs_id)
        REFERENCES price_optimization_inputs(id)
        ON DELETE CASCADE
);

