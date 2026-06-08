package com.emoagent.backend.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;
import jakarta.persistence.Table;

import java.time.Instant;

@Entity
@Table(name = "user_profiles")
public class UserProfile {
    @Id
    @Column(nullable = false, updatable = false)
    private String id;

    @Column(name = "user_id", nullable = false, unique = true)
    private String userId;

    @Lob
    @Column(name = "profile_data", nullable = false)
    private String profileData;

    @Column(name = "record_count", nullable = false)
    private Integer recordCount;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    @Column(name = "updated_at", nullable = false)
    private Instant updatedAt;

    protected UserProfile() {
    }

    public UserProfile(
            String id,
            String userId,
            String profileData,
            Integer recordCount,
            Instant createdAt,
            Instant updatedAt
    ) {
        this.id = id;
        this.userId = userId;
        this.profileData = profileData;
        this.recordCount = recordCount;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt;
    }

    public String getId() {
        return id;
    }

    public String getUserId() {
        return userId;
    }

    public String getProfileData() {
        return profileData;
    }

    public Integer getRecordCount() {
        return recordCount;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }

    public void setProfileData(String profileData) {
        this.profileData = profileData;
    }

    public void setRecordCount(Integer recordCount) {
        this.recordCount = recordCount;
    }

    public void setUpdatedAt(Instant updatedAt) {
        this.updatedAt = updatedAt;
    }
}
