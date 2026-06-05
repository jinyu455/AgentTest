package com.emoagent.backend.repository;

import com.emoagent.backend.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

//继承JpaRepository，id为主键，操作用户表
//自动实现按用户名查找用户，判断用户是否存在
public interface UserRepository extends JpaRepository<User, String> {
    Optional<User> findByUsername(String username);

    boolean existsByUsername(String username);
}
