CREATE DATABASE second_database;

GO
-- Use second_database;
USE second_database;
CREATE TABLE second_table(
    id INT PRIMARY KEY,
    username VARCHAR(30),
);
--- 
INSERT INTO second_table(id, username) VALUES(1, 'Samus');
INSERT INTO second_table(id, username) VALUES(2, 'Mario');
INSERT INTO second_table(id, username) VALUES(3, 'Luigi');
DELETE FROM second_table WHERE username='Luigi';
--
SELECT * FROM second_table;
DELETE FROM second_table WHERE username='Samus';
DELETE FROM second_table WHERE username='Mario';
DROP TABLE second_table;

-- MODIFY NAME to mario_database
ALTER DATABASE second_database MODIFY NAME = mario_database;
SELECT name FROM sys.databases;

USE mario_database;
CREATE TABLE characters(
	character_id INT PRIMARY KEY,
	name VARCHAR(30),
	homeland VARCHAR(60),
    favorite_color VARCHAR(60),
);

INSERT INTO characters(name, homeland, favorite_color) 
VALUES('Mario', 'Mushroom Kingdom', 'Red');
('Luigi', 'Mushroom Kingdom', 'Green')
('Peach', 'Mushroom Kingdom', 'Pink')
('Toadstool', 'Mushroom Kingdom', 'Red'),
('Bower', 'Mushroom Kingdom', 'Green'),
('Daisy', 'Sarasaland', 'Yellow'),
('Yoshi', 'Dinosaur Land', 'Green');

SELECT * FROM characters ORDER BY character_id;